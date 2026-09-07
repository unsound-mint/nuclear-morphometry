"""Cellpose segmentation backend (spec section 13.2).

Built against the installed Cellpose 4.2.1.1 (Cellpose-SAM), which is a
materially different API from the tissue-specific-model generation
(``model_type="nuclei"`` etc.) the build spec's phrasing assumes -- see
``docs/decisions/0001-cellpose-segmentation-backend.md`` for the full
investigation. Key consequences applied here:

- ``CellposeModel`` takes ``pretrained_model`` (a name or filesystem path)
  and an explicit ``device=torch.device(...)``, never the bare ``gpu=True``
  shortcut, so the selected device is auditable in provenance.
- ``model_type``/``diam_mean`` are dead parameters in this version and are
  never passed.
- ``eval()`` supports ``anisotropy`` for 3D, satisfying spec 13.2's
  "physical anisotropy passed to the model when the current API supports
  it" requirement.
- Cellpose's own model-name resolution silently falls back to its default
  model on an unrecognized name (logs a warning, does not raise) --
  this backend pre-validates the requested model name/path itself and
  raises before ever calling into Cellpose, so a config typo cannot
  silently substitute a different model.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from nuclear_morphometry.models import PhysicalSpacing, SegmentationResult
from nuclear_morphometry.segmentation.base import SegmenterUnavailableError, validate_label_image

logger = logging.getLogger(__name__)

# The library's own current foundation-model default (spec 11.1's "safe
# documented temporary development default"). Used only when the caller has
# explicitly opted into an unvalidated demo run -- see
# docs/decisions/0001-cellpose-segmentation-backend.md decision 3.
DEMO_DEFAULT_MODEL = "cpsam_v2"

_CELLPOSE_EVAL_DEFAULT_BATCH_SIZE = 8


def _resolve_model_name(model: str, *, allow_unvalidated_model: bool) -> str:
    """Resolve config.segmentation.model to a concrete Cellpose model name/path.

    Never delegates "is this a real model" to CellposeModel itself: that
    class logs a warning and silently substitutes its own default on an
    unrecognized name, which would violate this project's no-silent-fallback
    invariant (spec 13.2, 31). This function raises instead.
    """
    if model != "auto":
        if Path(model).exists():
            return model
        import cellpose.models as cp_models

        known = {*cp_models.MODEL_NAMES, *cp_models.get_user_models()}
        if model not in known:
            raise SegmenterUnavailableError(
                f"segmentation.model = {model!r} is neither an existing file path "
                f"nor a known Cellpose model name.\n\n"
                f"Known built-in models: {sorted(cp_models.MODEL_NAMES)}\n"
                f"Known user-trained models: {sorted(cp_models.get_user_models())}\n\n"
                f"Fix the config, or point it at a real weights file path."
            )
        return model

    if allow_unvalidated_model:
        logger.warning(
            'segmentation.model = "auto" resolved to the unvalidated demo default '
            "%r because this run explicitly allowed it (--allow-unvalidated-model). "
            "This model has not been validated against a reference mask set for this "
            "dataset (spec section 14). Do not use this for a final thesis run.",
            DEMO_DEFAULT_MODEL,
        )
        return DEMO_DEFAULT_MODEL

    raise SegmenterUnavailableError(
        'segmentation.model = "auto" requires a validated default model, and none has '
        "been selected yet for this project (spec section 11.1).\n\n"
        "Run segmentation validation to pick and accept a model:\n"
        "  nuclear-morphometry validate-segmentation --prediction <mask> --reference <mask>\n\n"
        "Then set segmentation.model to that model's name/path explicitly in the "
        "config. If you only need to exercise the pipeline on non-final data, rerun "
        "with --allow-unvalidated-model to use the library's current default "
        f"({DEMO_DEFAULT_MODEL!r}) unvalidated."
    )


def _resolve_device(device: Literal["cuda", "cpu"]) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise SegmenterUnavailableError(
            'segmentation.device = "cuda" but no CUDA device is available on this '
            "machine.\n\n"
            "Run `nuclear-morphometry doctor` to check GPU availability. Production runs "
            "must not silently fall back to CPU (spec section 13.2); switch to "
            'segmentation.device = "cpu" explicitly if that is genuinely intended '
            "(development/debugging only)."
        )
    return torch.device(device)


def _um_diameter_to_px(diameter_um: float, spacing: PhysicalSpacing) -> float | None:
    if diameter_um <= 0:
        return None
    return diameter_um / spacing.x_um


class CellposeSegmenter:
    """Cellpose-SAM backend. Loads its model once at construction (spec 13.2, 30.2)."""

    def __init__(
        self,
        *,
        model: str,
        device: Literal["cuda", "cpu"],
        diameter_um: float,
        use_anisotropy: bool,
        flow3d_smooth: float,
        batch_size: int,
        allow_unvalidated_model: bool = False,
    ) -> None:
        import cellpose.models as cp_models

        resolved_model = _resolve_model_name(model, allow_unvalidated_model=allow_unvalidated_model)
        torch_device = _resolve_device(device)

        self._model = cp_models.CellposeModel(pretrained_model=resolved_model, device=torch_device)
        self._resolved_model_name = resolved_model
        self._device = torch_device
        self._diameter_um = diameter_um
        self._use_anisotropy = use_anisotropy
        self._flow3d_smooth = flow3d_smooth
        self._configured_batch_size = batch_size
        self._oom_retries_last_call = 0

        import cellpose

        self._cellpose_version = cellpose.version

    def segment(
        self,
        image: NDArray[Any],
        spacing: PhysicalSpacing,
    ) -> SegmentationResult:
        do_3d = image.ndim == 3
        diameter = _um_diameter_to_px(self._diameter_um, spacing)
        anisotropy = (
            spacing.require_z(context="Cellpose 3D segmentation") / spacing.x_um
            if do_3d and self._use_anisotropy
            else None
        )
        starting_batch_size = (
            self._configured_batch_size
            if self._configured_batch_size > 0
            else _CELLPOSE_EVAL_DEFAULT_BATCH_SIZE
        )

        masks, batch_size_used, retries = self._eval_with_oom_retry(
            image,
            do_3d=do_3d,
            diameter=diameter,
            anisotropy=anisotropy,
            starting_batch_size=starting_batch_size,
        )
        self._oom_retries_last_call = retries

        labels = np.asarray(masks).astype(np.int32)
        validate_label_image(labels, expected_ndim=image.ndim)

        return SegmentationResult(
            labels=labels,
            backend="cellpose",
            model_id=self._resolved_model_name,
            backend_metadata={
                "cellpose_version": self._cellpose_version,
                "device": str(self._device),
                "diameter_px": diameter,
                "anisotropy": anisotropy,
                "flow3d_smooth": self._flow3d_smooth if do_3d else None,
                "batch_size_used": batch_size_used,
                "oom_retries": retries,
                "do_3D": do_3d,
            },
        )

    def _eval_with_oom_retry(
        self,
        image: NDArray[Any],
        *,
        do_3d: bool,
        diameter: float | None,
        anisotropy: float | None,
        starting_batch_size: int,
    ) -> tuple[NDArray[Any], int, int]:
        """Run model.eval, retrying once at a smaller batch size on CUDA OOM (spec 31).

        Never: falls back to CPU, resizes/downsamples the image, changes the model,
        or changes segmentation thresholds. Only the inference batch size shrinks.
        """
        batch_size = starting_batch_size
        retries = 0
        max_retries = 1

        while True:
            try:
                with torch.inference_mode():
                    masks, _flows, _styles = self._model.eval(
                        image,
                        batch_size=batch_size,
                        normalize=False,  # our own segmentation/normalize.py is authoritative
                        diameter=diameter,
                        do_3D=do_3d,
                        # Our images are single-channel ZYX with no channel axis;
                        # cellpose.transforms._convert_image_3d requires z_axis to
                        # be given explicitly for a plain ndim==3 volume (raises
                        # otherwise) -- verified empirically, this is not
                        # auto-detected the way the 2D no-channel case is.
                        z_axis=0 if do_3d else None,
                        anisotropy=anisotropy,
                        # Cellpose ships no type stubs; its own docstring allows
                        # int/float/list here, but pyright infers `int` from the
                        # untyped `flow3D_smooth=0` default. Real stub gap, not a
                        # correctness issue -- see the cast in segmentation/fixture.py
                        # for the same class of gap.
                        flow3D_smooth=cast("int", self._flow3d_smooth if do_3d else 0),
                    )
                return masks, batch_size, retries
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if retries >= max_retries or batch_size <= 1:
                    raise
                retries += 1
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "CUDA OOM during Cellpose inference; retrying with batch_size=%d "
                    "(retry %d/%d).",
                    batch_size,
                    retries,
                    max_retries,
                )
