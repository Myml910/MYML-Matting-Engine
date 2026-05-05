from __future__ import annotations

import os
import warnings
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.models.base import BaseMattingBackend, MattingResult


WHITE_THRESHOLD = 44
FOREGROUND_WHITE_DISTANCE = 72
FOREGROUND_MIN_SATURATION = 45
FOREGROUND_DISTANCE_FROM_BG = 3
DETAIL_DARK_VALUE = 120
DETAIL_VERY_DARK_VALUE = 70
DETAIL_CONTRAST_THRESHOLD = 18.0
THIN_GRADIENT_THRESHOLD = 16.0
THIN_LIGHT_GRADIENT_THRESHOLD = 10.0
THIN_VARIANCE_THRESHOLD = 12.0
THIN_NEIGHBOR_RADIUS = 2
RECOVERY_ALPHA_MIN = 20
RECOVERY_ALPHA_MAX = 150
RECOVERY_ALPHA_GAIN = 0.22
RECOVERY_WHITE_DISTANCE = 34
RECOVERY_MIN_SATURATION = 20
RECOVERY_CONTRAST_THRESHOLD = 10.0
RECOVERY_VARIANCE_THRESHOLD = 18.0
HOLE_WHITE_THRESHOLD = 12
HOLE_MAX_SATURATION = 12
HOLE_MIN_VALUE = 245
HOLE_MAX_STDDEV = 4.5
MIN_HOLE_AREA = 80
MAX_HOLE_AREA_RATIO = 0.006
HOLE_RING_RADIUS = 4
HOLE_MIN_FOREGROUND_RING_RATIO = 0.78
DECONTAMINATE_STRENGTH = 0.5
SOFTEN_EDGE_RADIUS = 2
SOFTEN_BLUR_SIGMA = 1.1
SOFTEN_BLEND = 0.55
SEMANTIC_FG_THRESHOLD = 218
SEMANTIC_UNKNOWN_THRESHOLD = 128
SEMANTIC_UNKNOWN_RELEASE_THRESHOLD = 96
GRAPHIC_MIN_AREA = 3
GRAPHIC_MAX_AREA_RATIO = 0.02
GRAPHIC_GRADIENT_THRESHOLD = 22.0
GRAPHIC_CONTRAST_THRESHOLD = 18.0
GRAPHIC_WHITE_DISTANCE = 42
LOCAL_REFINE_BLEND = 0.22
LOCAL_REFINE_DISAGREEMENT = 28
LOCAL_REFINE_MAX_ALPHA = 245
GAP_LOW_ALPHA_THRESHOLD = 32
GAP_MAX_AREA = 96
GAP_RING_RADIUS = 2
GAP_MIN_HIGH_ALPHA_RING_RATIO = 0.68
GAP_MIN_DETAIL_RING_RATIO = 0.12


_SEMANTIC_REMOVER: Any | None = None
_SEMANTIC_REMOVER_LOCK = Lock()


class WhiteBackgroundBackend(BaseMattingBackend):
    name = "whitebg"

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        rgb = image.convert("RGB")
        mask = remove_connected_white_background(rgb)
        rgba = apply_alpha_decontaminated(rgb, mask)
        return MattingResult(model_name=self.name, rgba=rgba, mask=mask)


def remove_connected_white_background(
    image: Image.Image,
    white_threshold: int = WHITE_THRESHOLD,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.size == 0:
        raise ValueError("Input image is empty")

    # 1. Build a white-background prior from near-white pixels connected to the image edges.
    distance_from_white = np.max(255 - rgb, axis=2)
    near_white = distance_from_white < white_threshold
    sure_background = _edge_connected_region(near_white)
    semantic_alpha = _semantic_alpha_or_empty(rgb)
    semantic_fg_seed = _semantic_foreground_seed(semantic_alpha)
    semantic_unknown = _semantic_unknown_region(semantic_alpha, semantic_fg_seed)
    sure_background = sure_background & ~semantic_fg_seed & ~semantic_unknown

    # 2. Build a trimap: sure background, sure foreground, and unknown transition/shallow regions.
    hole_background = _enclosed_white_holes(rgb, near_white, sure_background)
    sure_background = (sure_background | hole_background) & ~semantic_fg_seed & ~semantic_unknown
    graphic_fg_seed = _graphic_foreground_seed(rgb, near_white, sure_background)
    sure_foreground, thin_fg_seed = _sure_foreground(
        rgb,
        near_white,
        sure_background,
        semantic_fg_seed,
        graphic_fg_seed,
    )
    trimap = _build_trimap(sure_background, sure_foreground)

    # 3. Estimate soft alpha from the original image and trimap with pymatting.
    alpha = _estimate_alpha_with_pymatting(rgb, trimap)
    alpha, recovery_mask = _recover_foreground_alpha(rgb, alpha, near_white, sure_background)
    alpha_before_softening = alpha.copy()
    alpha = _soften_alpha_edges(alpha)
    whitebg_alpha = alpha.copy()
    alpha_before_local_refine = alpha.copy()
    alpha, uncertain_refine_region = _local_refine_alpha(rgb, alpha, semantic_alpha, sure_background, trimap)
    alpha, gap_repair_mask = _repair_small_alpha_gaps(rgb, alpha, sure_background)
    _write_debug_images(
        near_white,
        sure_background,
        sure_foreground,
        trimap,
        alpha,
        recovery_mask,
        thin_fg_seed,
        alpha_before_softening,
        whitebg_alpha,
        semantic_alpha,
        semantic_fg_seed,
        graphic_fg_seed,
        uncertain_refine_region,
        alpha_before_local_refine,
        gap_repair_mask,
    )

    return Image.fromarray(alpha, mode="L")


def apply_alpha_decontaminated(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)

    # 4. Reduce white background contamination on semi-transparent edge pixels.
    edge = (alpha > 0) & (alpha < 255)
    if np.any(edge):
        edge_alpha = np.clip(alpha[..., None] / 255.0, 0.05, 1.0)
        corrected = (rgb - 255.0 * (1.0 - edge_alpha)) / edge_alpha
        strength = (1.0 - edge_alpha) * DECONTAMINATE_STRENGTH
        rgb = np.where(edge[..., None], rgb * (1.0 - strength) + corrected * strength, rgb)

    rgba = np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), alpha.astype(np.uint8)))
    return Image.fromarray(rgba, mode="RGBA")


def _sure_foreground(
    rgb: np.ndarray,
    near_white: np.ndarray,
    sure_background: np.ndarray,
    semantic_fg_seed: np.ndarray,
    graphic_fg_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    distance_from_white = np.max(255 - rgb, axis=2)
    value = hsv[..., 2]
    local_contrast = _local_luma_contrast(rgb)
    local_variance = _local_luma_variance(rgb)
    gradient = _luma_gradient(rgb)
    clear_nonwhite = (
        (distance_from_white >= FOREGROUND_WHITE_DISTANCE)
        | ((hsv[..., 1] >= FOREGROUND_MIN_SATURATION) & ~near_white)
    )
    fine_detail = (
        (value <= DETAIL_VERY_DARK_VALUE)
        | ((value <= DETAIL_DARK_VALUE) & (local_contrast >= DETAIL_CONTRAST_THRESHOLD))
        | ((hsv[..., 1] >= FOREGROUND_MIN_SATURATION) & (local_contrast >= DETAIL_CONTRAST_THRESHOLD))
    )
    distance_from_background = cv2.distanceTransform((~sure_background).astype(np.uint8), cv2.DIST_L2, 3)
    interior_foreground = clear_nonwhite & (distance_from_background >= FOREGROUND_DISTANCE_FROM_BG)
    known_foreground_core = interior_foreground | fine_detail | semantic_fg_seed | graphic_fg_seed
    thin_fg_seed = _thin_foreground_seed(
        near_white=near_white,
        sure_background=sure_background,
        known_foreground_core=known_foreground_core,
        distance_from_white=distance_from_white,
        saturation=hsv[..., 1],
        value=value,
        gradient=gradient,
        local_contrast=local_contrast,
        local_variance=local_variance,
    )
    foreground = (interior_foreground | fine_detail | thin_fg_seed | semantic_fg_seed | graphic_fg_seed) & ~sure_background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    protected_detail = fine_detail & ~sure_background
    stable_foreground = cv2.erode(foreground.astype(np.uint8), kernel, iterations=1).astype(bool)
    return stable_foreground | protected_detail | thin_fg_seed | semantic_fg_seed | graphic_fg_seed, thin_fg_seed


def _thin_foreground_seed(
    near_white: np.ndarray,
    sure_background: np.ndarray,
    known_foreground_core: np.ndarray,
    distance_from_white: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    gradient: np.ndarray,
    local_contrast: np.ndarray,
    local_variance: np.ndarray,
) -> np.ndarray:
    kernel_size = THIN_NEIGHBOR_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    near_existing_fg = cv2.dilate(known_foreground_core.astype(np.uint8), kernel, iterations=1).astype(bool)

    dark_thin = (
        (value <= DETAIL_DARK_VALUE)
        & (gradient >= THIN_GRADIENT_THRESHOLD)
    )
    textured_light_edge = (
        ~near_white
        & (
            (gradient >= THIN_LIGHT_GRADIENT_THRESHOLD)
            | (local_contrast >= THIN_LIGHT_GRADIENT_THRESHOLD)
            | (local_variance >= THIN_VARIANCE_THRESHOLD)
        )
        & (
            (distance_from_white >= RECOVERY_WHITE_DISTANCE)
            | (saturation >= RECOVERY_MIN_SATURATION)
        )
    )
    seed = (dark_thin | textured_light_edge) & near_existing_fg & ~sure_background
    seed = seed & ~(near_white & (saturation < RECOVERY_MIN_SATURATION) & (local_variance < THIN_VARIANCE_THRESHOLD))
    return seed


def _build_trimap(sure_background: np.ndarray, sure_foreground: np.ndarray) -> np.ndarray:
    trimap = np.full(sure_background.shape, 128, dtype=np.uint8)
    trimap[sure_background] = 0
    trimap[sure_foreground & ~sure_background] = 255
    return trimap


def _semantic_alpha_or_empty(rgb: np.ndarray) -> np.ndarray:
    try:
        return _semantic_alpha(rgb)
    except Exception as exc:
        warnings.warn(f"Semantic foreground trimap protection unavailable; using whitebg trimap only. Reason: {exc}")
        return np.zeros(rgb.shape[:2], dtype=np.uint8)


def _semantic_foreground_seed(semantic_alpha: np.ndarray) -> np.ndarray:
    if not np.any(semantic_alpha):
        return np.zeros_like(semantic_alpha, dtype=bool)
    seed = semantic_alpha >= SEMANTIC_FG_THRESHOLD
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(seed.astype(np.uint8), kernel, iterations=1).astype(bool)


def _semantic_unknown_region(semantic_alpha: np.ndarray, semantic_fg_seed: np.ndarray) -> np.ndarray:
    if not np.any(semantic_alpha):
        return np.zeros_like(semantic_fg_seed, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    semantic_mid = semantic_alpha >= SEMANTIC_UNKNOWN_RELEASE_THRESHOLD
    semantic_high = semantic_alpha >= SEMANTIC_UNKNOWN_THRESHOLD
    near_semantic = cv2.dilate((semantic_fg_seed | semantic_high).astype(np.uint8), kernel, iterations=1).astype(bool)
    return semantic_mid & near_semantic & ~semantic_fg_seed


def _graphic_foreground_seed(
    rgb: np.ndarray,
    near_white: np.ndarray,
    sure_background: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    distance_from_white = np.max(255 - rgb, axis=2)
    gradient = _luma_gradient(rgb)
    contrast = _local_luma_contrast(rgb)
    raw_seed = (
        ~sure_background
        & (
            (distance_from_white >= GRAPHIC_WHITE_DISTANCE)
            | (hsv[..., 1] >= RECOVERY_MIN_SATURATION)
        )
        & (
            (gradient >= GRAPHIC_GRADIENT_THRESHOLD)
            | (contrast >= GRAPHIC_CONTRAST_THRESHOLD)
        )
    )
    raw_seed = raw_seed & ~(near_white & (hsv[..., 1] < RECOVERY_MIN_SATURATION))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_seed.astype(np.uint8), connectivity=8)
    max_area = max(GRAPHIC_MIN_AREA, int(rgb.shape[0] * rgb.shape[1] * GRAPHIC_MAX_AREA_RATIO))
    seed = np.zeros_like(raw_seed, dtype=bool)
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if GRAPHIC_MIN_AREA <= area <= max_area:
            seed[labels == label] = True
    return seed


def _estimate_alpha_with_pymatting(rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    try:
        from pymatting import estimate_alpha_cf
    except ImportError as exc:
        raise RuntimeError(
            "pymatting is required for white-background removal. Install it with "
            "`python -m pip install pymatting` or `python -m pip install -r requirements.txt`."
        ) from exc

    image_float = rgb.astype(np.float64) / 255.0
    trimap_float = trimap.astype(np.float64) / 255.0
    try:
        alpha = estimate_alpha_cf(image_float, trimap_float)
    except Exception as exc:
        raise RuntimeError("pymatting alpha estimation failed for the generated trimap.") from exc

    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[trimap == 0] = 0.0
    alpha[trimap == 255] = 1.0
    return (alpha * 255.0).astype(np.uint8)


def _recover_foreground_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    near_white: np.ndarray,
    sure_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    distance_from_white = np.max(255 - rgb, axis=2)
    local_contrast = _local_luma_contrast(rgb)
    local_variance = _local_luma_variance(rgb)

    alpha_candidate = (alpha >= RECOVERY_ALPHA_MIN) & (alpha <= RECOVERY_ALPHA_MAX)
    texture_evidence = (
        (local_contrast >= RECOVERY_CONTRAST_THRESHOLD)
        | (local_variance >= RECOVERY_VARIANCE_THRESHOLD)
    )
    color_evidence = (
        (distance_from_white >= RECOVERY_WHITE_DISTANCE)
        | (hsv[..., 1] >= RECOVERY_MIN_SATURATION)
    )

    recovery_mask = alpha_candidate & ~sure_background & (texture_evidence | color_evidence)
    recovery_mask = recovery_mask & ~(near_white & ~texture_evidence & (hsv[..., 1] < RECOVERY_MIN_SATURATION))

    recovered = alpha.copy()
    if np.any(recovery_mask):
        alpha_float = recovered.astype(np.float32)
        alpha_float[recovery_mask] = alpha_float[recovery_mask] + (255.0 - alpha_float[recovery_mask]) * RECOVERY_ALPHA_GAIN
        recovered = np.clip(alpha_float, 0, 255).astype(np.uint8)

        smoothed = cv2.GaussianBlur(recovered, (3, 3), 0)
        blend_region = cv2.dilate(recovery_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(
            bool
        )
        recovered[blend_region] = ((recovered[blend_region].astype(np.uint16) + smoothed[blend_region]) // 2).astype(
            np.uint8
        )
        recovered[sure_background] = 0

    return recovered, recovery_mask


def _soften_alpha_edges(alpha: np.ndarray) -> np.ndarray:
    semi_transparent = (alpha > 0) & (alpha < 255)
    if not np.any(semi_transparent):
        return alpha

    kernel_size = SOFTEN_EDGE_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    edge_band = cv2.dilate(semi_transparent.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_band = edge_band & (alpha > 0) & (alpha < 255)

    blurred = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), SOFTEN_BLUR_SIGMA)
    alpha_float = alpha.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    softened = alpha.copy()
    mixed = alpha_float * (1.0 - SOFTEN_BLEND) + blurred_float * SOFTEN_BLEND
    softened[edge_band] = np.clip(mixed[edge_band], 0, 255).astype(np.uint8)

    softened[alpha == 0] = 0
    softened[alpha == 255] = 255
    return softened


def _local_refine_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    semantic_alpha: np.ndarray,
    sure_background: np.ndarray,
    trimap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(semantic_alpha):
        return alpha, np.zeros_like(alpha, dtype=bool)

    gradient = _luma_gradient(rgb)
    variance = _local_luma_variance(rgb)
    semi_alpha = (alpha > 0) & (alpha < 255)
    disagreement = semantic_alpha.astype(np.int16) - alpha.astype(np.int16)
    detail_evidence = (gradient >= THIN_LIGHT_GRADIENT_THRESHOLD) | (variance >= THIN_VARIANCE_THRESHOLD)
    trimap_uncertain = trimap == 128
    semantic_support = semantic_alpha >= SEMANTIC_UNKNOWN_THRESHOLD
    refine_region = (
        ~sure_background
        & (alpha < LOCAL_REFINE_MAX_ALPHA)
        & (disagreement >= LOCAL_REFINE_DISAGREEMENT)
        & semantic_support
        & (
            semi_alpha
            | trimap_uncertain
            | detail_evidence
        )
    )
    if not np.any(refine_region):
        return alpha, refine_region

    refined = alpha.astype(np.float32)
    semantic_float = semantic_alpha.astype(np.float32)
    confidence = np.clip((semantic_float - alpha.astype(np.float32)) / 255.0, 0.0, 1.0)
    local_blend = LOCAL_REFINE_BLEND * (0.35 + confidence)
    refined[refine_region] = (
        refined[refine_region] * (1.0 - local_blend[refine_region])
        + semantic_float[refine_region] * local_blend[refine_region]
    )
    smoothed = cv2.GaussianBlur(np.clip(refined, 0, 255).astype(np.uint8), (3, 3), 0)
    blend_band = cv2.dilate(refine_region.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    refined_u8 = np.clip(refined, 0, 255).astype(np.uint8)
    refined_u8[blend_band] = ((refined_u8[blend_band].astype(np.uint16) + smoothed[blend_band]) // 2).astype(np.uint8)
    refined_u8[sure_background] = 0
    refined_u8[alpha == 255] = 255
    return refined_u8, refine_region


def _repair_small_alpha_gaps(
    rgb: np.ndarray,
    alpha: np.ndarray,
    sure_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low_alpha = (alpha < GAP_LOW_ALPHA_THRESHOLD) & ~sure_background
    low_alpha = low_alpha & ~_edge_connected_region(low_alpha)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(low_alpha.astype(np.uint8), connectivity=8)
    if component_count <= 1:
        return alpha, np.zeros_like(alpha, dtype=bool)

    gradient = _luma_gradient(rgb)
    variance = _local_luma_variance(rgb)
    detail = (gradient >= THIN_LIGHT_GRADIENT_THRESHOLD) | (variance >= THIN_VARIANCE_THRESHOLD)
    kernel_size = GAP_RING_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    repaired = alpha.copy()
    repair_mask = np.zeros_like(alpha, dtype=bool)
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > GAP_MAX_AREA:
            continue

        component = labels == label
        if _touches_edge(component):
            continue

        dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring = ring & ~sure_background
        if not np.any(ring):
            continue

        high_alpha_ring = ring & (alpha >= 220)
        high_alpha_ratio = np.count_nonzero(high_alpha_ring) / np.count_nonzero(ring)
        detail_ratio = np.count_nonzero(ring & detail) / np.count_nonzero(ring)
        if high_alpha_ratio < GAP_MIN_HIGH_ALPHA_RING_RATIO and detail_ratio < GAP_MIN_DETAIL_RING_RATIO:
            continue

        fill_source = alpha[high_alpha_ring]
        if fill_source.size == 0:
            fill_source = alpha[ring]
        fill_value = int(np.clip(np.median(fill_source), 180, 255))
        repaired[component] = np.maximum(repaired[component], fill_value)
        repair_mask[component] = True

    if np.any(repair_mask):
        smoothed = cv2.GaussianBlur(repaired, (3, 3), 0)
        blend_band = cv2.dilate(repair_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        repaired[blend_band] = np.maximum(repaired[blend_band], smoothed[blend_band])
        repaired[sure_background] = 0
    return repaired, repair_mask


def _apply_semantic_foreground_protection(
    rgb: np.ndarray,
    whitebg_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        semantic_alpha = _semantic_alpha(rgb)
    except Exception as exc:
        warnings.warn(f"Semantic foreground protection unavailable; using whitebg alpha only. Reason: {exc}")
        empty = np.zeros_like(whitebg_alpha, dtype=np.uint8)
        return whitebg_alpha, empty, empty

    semantic_core = semantic_alpha >= SEMANTIC_PROTECTION_THRESHOLD
    if not np.any(semantic_core):
        empty = np.zeros_like(whitebg_alpha, dtype=np.uint8)
        return whitebg_alpha, semantic_alpha, empty

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SEMANTIC_PROTECTION_FEATHER, SEMANTIC_PROTECTION_FEATHER),
    )
    softened_core = cv2.GaussianBlur((semantic_core.astype(np.uint8) * 255), (0, 0), sigmaX=1.2)
    feathered_core = cv2.dilate(softened_core, kernel, iterations=1)
    semantic_protection = np.minimum(semantic_alpha, feathered_core).astype(np.uint8)
    semantic_protection[semantic_alpha < SEMANTIC_PROTECTION_THRESHOLD] = 0

    final_alpha = np.maximum(whitebg_alpha, semantic_protection).astype(np.uint8)
    return final_alpha, semantic_alpha, semantic_protection


def _semantic_alpha(rgb: np.ndarray) -> np.ndarray:
    remover = _get_semantic_remover()
    image = Image.fromarray(rgb, mode="RGB")
    try:
        output = remover.process(image, type="rgba")
    except TypeError:
        output = remover.process(image)

    alpha = _extract_semantic_alpha(output)
    if alpha.shape != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return alpha


def _get_semantic_remover() -> Any:
    global _SEMANTIC_REMOVER
    if _SEMANTIC_REMOVER is not None:
        return _SEMANTIC_REMOVER

    with _SEMANTIC_REMOVER_LOCK:
        if _SEMANTIC_REMOVER is not None:
            return _SEMANTIC_REMOVER

        from transparent_background import Remover

        _SEMANTIC_REMOVER = Remover()
        return _SEMANTIC_REMOVER


def _extract_semantic_alpha(output: Any) -> np.ndarray:
    if isinstance(output, Image.Image):
        output_image = output
        if output_image.mode == "RGBA":
            return np.asarray(output_image.getchannel("A"), dtype=np.uint8)
        if output_image.mode == "LA":
            return np.asarray(output_image.getchannel("A"), dtype=np.uint8)
        if output_image.mode == "L":
            return np.asarray(output_image, dtype=np.uint8)
        raise RuntimeError(f"Expected semantic RGBA or mask output, got mode {output_image.mode}")

    if isinstance(output, np.ndarray):
        array = output
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array[..., -1]
        raise RuntimeError(f"Unexpected semantic output shape: {array.shape}")

    raise RuntimeError(f"Unexpected semantic output type: {type(output).__name__}")


def _enclosed_white_holes(rgb: np.ndarray, near_white: np.ndarray, edge_background: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    distance_from_white = np.max(255 - rgb, axis=2)
    strict_white = (
        (distance_from_white < HOLE_WHITE_THRESHOLD)
        & (hsv[..., 1] <= HOLE_MAX_SATURATION)
        & (hsv[..., 2] >= HOLE_MIN_VALUE)
        & ~edge_background
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(strict_white.astype(np.uint8), connectivity=8)
    max_area = max(MIN_HOLE_AREA, int(rgb.shape[0] * rgb.shape[1] * MAX_HOLE_AREA_RATIO))
    holes = np.zeros_like(edge_background, dtype=bool)
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_HOLE_AREA or area > max_area:
            continue

        component = labels == label
        if _touches_edge(component):
            continue
        if _component_stddev(rgb, component) > HOLE_MAX_STDDEV:
            continue
        if _foreground_ring_ratio(component, near_white, edge_background) < HOLE_MIN_FOREGROUND_RING_RATIO:
            continue

        holes[component] = True
    return holes


def _component_stddev(rgb: np.ndarray, component: np.ndarray) -> float:
    pixels = rgb[component].astype(np.float32)
    if pixels.size == 0:
        return 255.0
    return float(np.mean(np.std(pixels, axis=0)))


def _local_luma_contrast(rgb: np.ndarray) -> np.ndarray:
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_min = cv2.erode(luma, np.ones((3, 3), dtype=np.uint8), iterations=1)
    local_max = cv2.dilate(luma, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return local_max - local_min


def _local_luma_variance(rgb: np.ndarray) -> np.ndarray:
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(luma, (5, 5))
    mean_sq = cv2.blur(luma * luma, (5, 5))
    return np.maximum(mean_sq - mean * mean, 0.0)


def _luma_gradient(rgb: np.ndarray) -> np.ndarray:
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def _foreground_ring_ratio(component: np.ndarray, near_white: np.ndarray, edge_background: np.ndarray) -> float:
    kernel_size = HOLE_RING_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~component
    ring = ring & ~edge_background
    if not np.any(ring):
        return 0.0
    foreground_ring = ring & ~near_white
    return float(np.count_nonzero(foreground_ring) / np.count_nonzero(ring))


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _write_debug_images(
    near_white: np.ndarray,
    sure_background: np.ndarray,
    sure_foreground: np.ndarray,
    trimap: np.ndarray,
    final_alpha: np.ndarray,
    recovery_mask: np.ndarray,
    thin_fg_seed: np.ndarray,
    alpha_before_softening: np.ndarray,
    whitebg_alpha: np.ndarray,
    semantic_alpha: np.ndarray,
    semantic_fg_seed: np.ndarray,
    graphic_fg_seed: np.ndarray,
    uncertain_refine_region: np.ndarray,
    alpha_before_local_refine: np.ndarray,
    gap_repair_mask: np.ndarray,
) -> None:
    debug_dir = os.getenv("WHITEBG_DEBUG_DIR")
    if not debug_dir:
        return

    output_dir = Path(debug_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((near_white.astype(np.uint8) * 255), mode="L").save(output_dir / "debug_near_white.png")
    Image.fromarray((sure_background.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_sure_background.png"
    )
    Image.fromarray((sure_foreground.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_sure_foreground.png"
    )
    Image.fromarray(trimap, mode="L").save(output_dir / "debug_trimap.png")
    Image.fromarray(alpha_before_softening, mode="L").save(output_dir / "debug_alpha_before_softening.png")
    Image.fromarray(whitebg_alpha, mode="L").save(output_dir / "debug_whitebg_alpha.png")
    Image.fromarray(semantic_alpha, mode="L").save(output_dir / "debug_semantic_alpha.png")
    Image.fromarray((semantic_fg_seed.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_semantic_fg_seed.png"
    )
    Image.fromarray((graphic_fg_seed.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_graphic_fg_seed.png"
    )
    Image.fromarray((sure_foreground.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_combined_sure_foreground.png"
    )
    Image.fromarray((uncertain_refine_region.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_uncertain_refine_region.png"
    )
    Image.fromarray(alpha_before_local_refine, mode="L").save(output_dir / "debug_alpha_before_local_refine.png")
    Image.fromarray(final_alpha, mode="L").save(output_dir / "debug_final_alpha.png")
    Image.fromarray(final_alpha, mode="L").save(output_dir / "debug_alpha_after_local_refine.png")
    Image.fromarray(final_alpha, mode="L").save(output_dir / "debug_alpha_after_softening.png")
    Image.fromarray((recovery_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_recovery_mask.png"
    )
    Image.fromarray((thin_fg_seed.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_thin_fg_seed.png"
    )
    Image.fromarray((gap_repair_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "debug_gap_repair_mask.png"
    )


def _edge_connected_region(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add_if_background(x: int, y: int) -> None:
        if mask[y, x] and not visited[y, x]:
            visited[y, x] = True
            queue.append((x, y))

    for x in range(width):
        add_if_background(x, 0)
        add_if_background(x, height - 1)
    for y in range(height):
        add_if_background(0, y)
        add_if_background(width - 1, y)

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                add_if_background(nx, ny)

    return visited
