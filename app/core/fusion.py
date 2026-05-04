from __future__ import annotations

import cv2
import numpy as np


def fuse_inspyrenet_ben2_edge(
    inspy_alpha: np.ndarray,
    ben2_alpha: np.ndarray,
    edge_radius: int = 8,
    blend: float = 1.0,
) -> np.ndarray:
    if inspy_alpha.ndim != 2:
        raise ValueError("inspy_alpha must be a single-channel uint8 alpha mask")

    inspy = _to_uint8_alpha(inspy_alpha)
    ben2 = _to_uint8_alpha(ben2_alpha)

    if ben2.shape != inspy.shape:
        target_width = inspy.shape[1]
        target_height = inspy.shape[0]
        ben2 = cv2.resize(ben2, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    if edge_radius <= 0:
        return inspy.copy()

    blend = max(0.0, min(1.0, float(blend)))
    foreground = inspy >= 128
    kernel_size = edge_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground_u8 = foreground.astype(np.uint8)
    dilated = cv2.dilate(foreground_u8, kernel, iterations=1)
    eroded = cv2.erode(foreground_u8, kernel, iterations=1)
    edge_band = (dilated - eroded).astype(bool)

    final = inspy.copy()
    if blend >= 1.0:
        final[edge_band] = ben2[edge_band]
    elif blend > 0.0:
        fused = inspy.astype(np.float32) * (1.0 - blend) + ben2.astype(np.float32) * blend
        final[edge_band] = np.clip(fused[edge_band], 0, 255).astype(np.uint8)

    return final


def _to_uint8_alpha(alpha: np.ndarray) -> np.ndarray:
    values = np.asarray(alpha)
    if values.ndim == 3:
        values = values[..., -1]
    if values.ndim != 2:
        raise ValueError("alpha mask must be a 2D array or an array with alpha in the last channel")
    if values.dtype == np.uint8:
        return values.copy()
    if np.issubdtype(values.dtype, np.floating) and values.size and values.max() <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)
