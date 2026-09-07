"""Interactive napari QC viewer (spec section 23).

The only module in this package allowed to import napari or Qt (see
AGENTS.md: "the computational core must not depend on napari or Qt") --
every piece of logic that does not itself need a live napari ``Viewer`` is
a plain function below so it can be (and is, in
``tests/integration/test_qc_viewer.py``) unit-tested without one. Callers
outside this module (``cli.py``) must import it lazily, inside the ``qc``
command body, since napari/Qt are the optional ``gui`` dependency group --
importing this module unconditionally would make every ``dayana-nuclei``
invocation require them.

Never mutates the raw source image or the saved production mask: the
labels array handed to napari is a plain in-memory copy from
``io.masks.load_label_mask``, the ``Labels``/``Points`` layers are
uneditable (``layer.editable = False``, so an accidental brush stroke or
polygon edit cannot corrupt the in-memory display, let alone anything on
disk), and every manual tag is written through
``qc.annotations.save_annotation`` into ``run_dir/qc/annotations.json`` --
never back into ``nuclei.parquet`` or ``masks/`` (see that module's own
docstring for the same non-mutation guarantee, which this reuses).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import napari
import numpy as np
import polars as pl
from magicgui.widgets import ComboBox, Container, Label, PushButton
from napari.layers import Image as ImageLayer
from napari.layers import Labels as LabelsLayer
from napari.layers import Points as PointsLayer
from numpy.typing import NDArray
from skimage.measure import regionprops

from dayana_nuclei.config import Config, load_config
from dayana_nuclei.io.images import load_channel_volume
from dayana_nuclei.io.manifest import resolve_image_sources
from dayana_nuclei.io.masks import load_label_mask, mask_path_for
from dayana_nuclei.models import ImageSource
from dayana_nuclei.qc.annotations import ManualTag, load_annotations, save_annotation

# Keyboard shortcuts for manual tags (spec 23). Checked empirically against
# napari 0.9.1's default Viewer/Labels/Image keymaps: none of g/d/m/s/o are
# bound by anything else at the Viewer level, so a plain single-letter
# viewer.bind_key is safe regardless of which layer is active.
_TAG_KEYS: dict[str, ManualTag] = {
    "g": "good",
    "d": "debris",
    "m": "merge",
    "s": "split",
    "o": "other",
}
_TAG_COLORS: dict[ManualTag, str] = {
    "good": "green",
    "debris": "red",
    "merge": "orange",
    "split": "blue",
    "other": "gray",
}

# Priority-ordered measurement columns to show for a selected object, when
# present and non-null on that row -- covers both 2D and 3D nuclei.parquet
# shapes (only one half of these is ever populated for a given run's mode)
# without hardcoding which mode is active.
_DISPLAY_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "area_um2",
    "perimeter_um",
    "circularity",
    "solidity",
    "eccentricity",
    "volume_um3",
    "surface_area_um2",
    "sphericity",
    "mean_intensity",
    "integrated_intensity",
    "qc_border",
    "qc_exclusion_reason",
)


def _add_image_layer(viewer: napari.Viewer, data: NDArray[Any], **kwargs: Any) -> ImageLayer:
    # add_image can return a list of layers when given a list of arrays; we
    # always pass one array, so the result is always one Image at runtime.
    return cast("ImageLayer", viewer.add_image(data, **kwargs))


def _add_labels_layer(
    viewer: napari.Viewer, data: NDArray[np.integer[Any]], **kwargs: Any
) -> LabelsLayer:
    # add_labels/add_points are attached to Viewer dynamically -- napari
    # generates one add_<layer type> method per registered layer type at
    # import time (napari.utils._register.create_func) -- which napari's
    # own (py.typed-less) source gives static analysis nothing to see. Same
    # class of stub gap as skimage's mark_boundaries in qc/report.py.
    return viewer.add_labels(data, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]


def _add_points_layer(
    viewer: napari.Viewer, data: NDArray[np.float64], **kwargs: Any
) -> PointsLayer:
    return viewer.add_points(data, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]


def _selected_object_number(value: Any) -> int | None:
    """Napari's ``Layer.get_value`` for a Labels layer returns the raw label
    at a position, or ``None``/0 for background/off-canvas -- both of which
    mean "nothing selected" for this viewer's purposes."""
    if value is None:
        return None
    value_int = int(value)
    return value_int if value_int > 0 else None


def _format_measurement_text(
    nuclei_df: pl.DataFrame, image_id: str, object_number: int | None, current_tag: ManualTag | None
) -> str:
    """Display text for the dock widget's info label (spec 23: "display its
    object number and key measurements")."""
    if object_number is None:
        return "No object selected. Click a nucleus in the labels layer."

    rows = nuclei_df.filter(
        (pl.col("image_id") == image_id) & (pl.col("object_number") == object_number)
    )
    if rows.height == 0:
        return f"Object {object_number}: not present in nuclei.parquet (stale selection?)."

    row = rows.row(0, named=True)
    lines = [f"Object {object_number}", f"tag: {current_tag or 'none'}"]
    for column in _DISPLAY_MEASUREMENT_COLUMNS:
        value = row.get(column)
        if value is None:
            continue
        lines.append(f"{column}: {value:.4g}" if isinstance(value, float) else f"{column}: {value}")
    return "\n".join(lines)


def _annotation_overlay_data(
    run_dir: Path, image_id: str, labels: NDArray[np.integer[Any]]
) -> tuple[NDArray[np.float64], list[ManualTag]]:
    """Centroid + tag for every object in ``labels`` that already has a
    saved manual annotation for ``image_id`` -- backs the "QC
    overlay/annotations" requirement and "existing annotations must reload"
    (spec 23). Pure numpy/skimage: works identically for 2D and 3D labels
    since ``regionprops.centroid`` is already (row, col) or (z, row, col).
    An annotated object_number no longer present in this particular mask
    (e.g. after a re-run with different segmentation) is silently skipped
    rather than raising -- the annotation itself is untouched either way.
    """
    annotations = load_annotations(run_dir)
    relevant: dict[int, ManualTag] = {
        object_number: entry.tag
        for (this_image_id, object_number), entry in annotations.items()
        if this_image_id == image_id
    }
    if not relevant:
        return np.empty((0, labels.ndim), dtype=np.float64), []

    centroids = {
        prop.label: prop.centroid for prop in regionprops(labels) if prop.label in relevant
    }
    points: list[tuple[float, ...]] = []
    tags: list[ManualTag] = []
    for object_number, tag in relevant.items():
        centroid = centroids.get(object_number)
        if centroid is None:
            continue
        points.append(centroid)
        tags.append(tag)
    if not points:
        return np.empty((0, labels.ndim), dtype=np.float64), []
    return np.array(points, dtype=np.float64), tags


@dataclass
class _RunContext:
    run_dir: Path
    config: Config
    fields_df: pl.DataFrame
    nuclei_df: pl.DataFrame
    sources: dict[str, dict[str, ImageSource]]
    image_ids: tuple[str, ...]


def _load_run_context(run_dir: Path) -> _RunContext:
    nuclei_path = run_dir / "nuclei.parquet"
    fields_path = run_dir / "fields.parquet"
    manifest_path = run_dir / "manifest.parquet"
    if not (nuclei_path.exists() and fields_path.exists() and manifest_path.exists()):
        raise FileNotFoundError(
            f"{run_dir} does not look like a finalized dayana-nuclei run directory (missing "
            f"nuclei.parquet/fields.parquet/manifest.parquet). Run `dayana-nuclei run` (or "
            f"resume/finalize an interrupted run) first."
        )
    config, _ = load_config(run_dir / "config.toml")
    if not config.output.save_masks:
        raise ValueError(
            f"{run_dir}'s config has output.save_masks = false; the QC viewer requires saved "
            f"label masks (spec 23) and this run has none."
        )

    fields_df = pl.read_parquet(fields_path)
    nuclei_df = pl.read_parquet(nuclei_path)
    manifest_df = pl.read_parquet(manifest_path)
    sources = resolve_image_sources(
        manifest_df,
        hoechst_channel=config.input.hoechst_channel,
        additional_channels=config.input.additional_channels,
    )
    # Only fields that actually completed (spec 33: a failed field never
    # reaches fields.parquet) and still resolve in the manifest are viewable.
    image_ids = tuple(sorted(set(fields_df["image_id"].to_list()) & set(sources.keys())))
    if not image_ids:
        raise ValueError(f"{run_dir} has no successfully analyzed fields to view.")

    return _RunContext(
        run_dir=run_dir,
        config=config,
        fields_df=fields_df,
        nuclei_df=nuclei_df,
        sources=sources,
        image_ids=image_ids,
    )


@dataclass
class _FieldArrays:
    hoechst: NDArray[Any]
    labels: NDArray[np.integer[Any]]
    channels: dict[str, NDArray[Any]] = field(default_factory=dict)


def _load_field_arrays(ctx: _RunContext, image_id: str) -> _FieldArrays:
    channels = ctx.sources[image_id]
    hoechst_source = channels[ctx.config.input.hoechst_channel]
    volume = load_channel_volume(
        hoechst_source,
        mode=ctx.config.analysis.mode,
        projection=ctx.config.analysis.projection,
        specific_plane=ctx.config.analysis.specific_plane,
    )

    mask_path = mask_path_for(ctx.run_dir, image_id)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"No saved mask for {image_id!r} at {mask_path} -- was this field processed with "
            f"output.save_masks = true?"
        )
    labels, _mask_axes = load_label_mask(mask_path)
    if labels.shape != volume.data.shape:
        raise ValueError(
            f"{image_id!r}: mask shape {labels.shape} does not match the raw image shape "
            f"{volume.data.shape}. The saved mask may be stale for this source file."
        )

    extra_channels: dict[str, NDArray[Any]] = {}
    for channel_name in ctx.config.input.additional_channels:
        source = channels.get(channel_name)
        if source is None:
            continue  # not present for this particular field (spec 25.2: "if present")
        channel_volume = load_channel_volume(
            source,
            mode=ctx.config.analysis.mode,
            projection=ctx.config.analysis.projection,
            specific_plane=ctx.config.analysis.specific_plane,
        )
        if channel_volume.data.shape == volume.data.shape:
            extra_channels[channel_name] = channel_volume.data

    return _FieldArrays(hoechst=volume.data, labels=labels, channels=extra_channels)


class QCViewer:
    """Live state for one napari QC session against one run directory.

    Constructed by ``build_qc_viewer``, which wires this instance's public
    methods to the dock widget's controls and the viewer's keyboard
    shortcuts. Holds no logic that depends on a live Qt event loop itself
    (napari's ``Viewer(show=False)`` and layer objects work without one),
    which is what makes this fully exercisable in a headless test.
    """

    def __init__(self, run_dir: Path, viewer: napari.Viewer, ctx: _RunContext) -> None:
        self.run_dir = run_dir
        self.viewer = viewer
        self.image_id: str = ctx.image_ids[0]
        self.object_number: int | None = None
        self.info_label: Label | None = None

        self._ctx = ctx
        self._labels: NDArray[np.integer[Any]] | None = None
        self._hoechst_layer: ImageLayer | None = None
        self._labels_layer: LabelsLayer | None = None
        self._channel_layers: dict[str, ImageLayer] = {}
        self._annotation_layer: PointsLayer | None = None

    def switch_field(self, image_id: str) -> None:
        """Load a different field's layers, replacing the current ones.
        Never touches the source file or mask on disk -- only reads them."""
        if image_id not in self._ctx.image_ids:
            raise ValueError(
                f"{image_id!r} is not one of this run's analyzed fields: {self._ctx.image_ids}"
            )
        arrays = _load_field_arrays(self._ctx, image_id)

        for layer in (
            self._hoechst_layer,
            self._labels_layer,
            self._annotation_layer,
            *self._channel_layers.values(),
        ):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._channel_layers = {}

        self.image_id = image_id
        self._labels = arrays.labels
        self.object_number = None

        self._hoechst_layer = _add_image_layer(self.viewer, arrays.hoechst, name="hoechst")
        self._labels_layer = _add_labels_layer(self.viewer, arrays.labels, name="labels")
        # Never let a stray brush/fill/polygon action alter the displayed
        # copy (spec 23: "never modify the raw image or production mask
        # in-place") -- editing is entirely unnecessary for this viewer's
        # click-to-select workflow.
        self._labels_layer.editable = False
        self._labels_layer.mouse_drag_callbacks.append(self._on_click)

        for channel_name, data in arrays.channels.items():
            self._channel_layers[channel_name] = _add_image_layer(
                self.viewer, data, name=channel_name, visible=False
            )

        self._refresh_annotation_overlay()
        self._refresh_info_label()

    def _on_click(self, layer: LabelsLayer, event: Any) -> None:
        value = layer.get_value(
            event.position,
            world=True,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
        )
        self.select_object(_selected_object_number(value))

    def select_object(self, object_number: int | None) -> None:
        self.object_number = object_number
        self._refresh_info_label()

    def _current_tag(self) -> ManualTag | None:
        if self.object_number is None:
            return None
        entry = load_annotations(self.run_dir).get((self.image_id, self.object_number))
        return entry.tag if entry is not None else None

    def _refresh_info_label(self) -> None:
        if self.info_label is not None:
            self.info_label.value = _format_measurement_text(
                self._ctx.nuclei_df, self.image_id, self.object_number, self._current_tag()
            )

    def _refresh_annotation_overlay(self) -> None:
        assert self._labels is not None
        if self._annotation_layer is not None and self._annotation_layer in self.viewer.layers:
            self.viewer.layers.remove(self._annotation_layer)

        points, tags = _annotation_overlay_data(self.run_dir, self.image_id, self._labels)
        if points.shape[0] == 0:
            self._annotation_layer = _add_points_layer(
                self.viewer, points, name="qc annotations", ndim=self._labels.ndim
            )
        else:
            colors = [_TAG_COLORS[tag] for tag in tags]
            self._annotation_layer = _add_points_layer(
                self.viewer,
                points,
                name="qc annotations",
                face_color=colors,
                # Marker size from the YX (last two) axes only: for a 3D ZYX
                # stack, the Z depth is often much smaller than the field of
                # view (e.g. 10 planes), and min(shape) over all axes would
                # make markers nearly invisible.
                size=max(2, min(self._labels.shape[-2:]) // 20),
                properties={"tag": tags},
                text={"string": "{tag}", "color": "white", "anchor": "upper_left"},
            )
        # Manual tags are recorded by clicking the labels layer, never by
        # dragging these markers around.
        self._annotation_layer.editable = False

    def apply_tag(self, tag: ManualTag) -> None:
        """Save ``tag`` for the currently selected object (spec 23's manual
        tags: good/debris/merge/split/other). A no-op (with an info-label
        message) if nothing is selected, never a crash from a stray
        keypress or dock-button click before a nucleus was picked."""
        if self.object_number is None:
            if self.info_label is not None:
                self.info_label.value = "Select an object (click a nucleus) before tagging."
            return
        save_annotation(
            self.run_dir, image_id=self.image_id, object_number=self.object_number, tag=tag
        )
        self._refresh_annotation_overlay()
        self._refresh_info_label()


def build_qc_viewer(run_dir: Path, *, image_id: str | None = None) -> QCViewer:
    """Construct a napari QC viewer for ``run_dir`` (spec section 23), with
    ``show=False`` -- the caller decides whether/when to enter napari's
    blocking Qt event loop (see ``launch_qc_viewer``). Returns the
    ``QCViewer`` wrapper (``.viewer`` is the underlying ``napari.Viewer``),
    fully testable without that event loop."""
    ctx = _load_run_context(run_dir)
    initial_image_id = image_id if image_id is not None else ctx.image_ids[0]
    if initial_image_id not in ctx.image_ids:
        raise ValueError(
            f"{initial_image_id!r} is not one of this run's analyzed fields: {ctx.image_ids}"
        )

    viewer = napari.Viewer(show=False, title=f"dayana-nuclei QC: {run_dir.name}")
    qc_viewer = QCViewer(run_dir, viewer, ctx)

    info_label = Label(value="No object selected.")
    qc_viewer.info_label = info_label

    field_combo = ComboBox(label="Field", choices=ctx.image_ids, value=initial_image_id)
    field_combo.changed.connect(qc_viewer.switch_field)

    def _make_tag_handler(tag: ManualTag) -> Callable[..., None]:
        # A plain closure, not a lambda with tag=tag as a default value:
        # pyright infers a lambda's un-annotated default-parameter type from
        # the default expression the same way it infers `def f(x=1)` as
        # `x: int` rather than `x: Literal[1]` -- that widening would lose
        # ManualTag's Literal type here too. bind_key calls back with the
        # Viewer; the button's `changed` event calls back with the new
        # checked state; neither is used, hence *_args.
        def handler(*_args: Any) -> None:
            qc_viewer.apply_tag(tag)

        return handler

    tag_buttons = []
    for key, tag in _TAG_KEYS.items():
        button = PushButton(text=f"{tag.capitalize()} ({key})")
        button.changed.connect(_make_tag_handler(tag))
        tag_buttons.append(button)
        viewer.bind_key(key, _make_tag_handler(tag))

    dock = Container(widgets=[field_combo, info_label, *tag_buttons])
    viewer.window.add_dock_widget(dock, name="QC annotations", area="right")

    qc_viewer.switch_field(initial_image_id)
    return qc_viewer


def launch_qc_viewer(run_dir: Path, *, image_id: str | None = None) -> None:
    """Build the viewer and enter napari's blocking event loop -- the real
    ``dayana-nuclei qc`` entry point. Never called by tests (see
    ``build_qc_viewer`` for the headlessly-testable half)."""
    build_qc_viewer(run_dir, image_id=image_id)
    napari.run()
