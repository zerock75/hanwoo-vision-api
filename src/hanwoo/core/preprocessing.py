"""
Image preprocessing module for background removal and alignment.
Uses rembg library with GPU support and u2net model for background removal.
Includes automatic rotation correction and smart cropping.
"""

import sys
import os
import warnings
from pathlib import Path

from .config import U2NET_HOME

os.environ.setdefault("U2NET_HOME", str(U2NET_HOME))
os.environ.setdefault("MODEL_CHECKSUM_DISABLED", "1")

# Add rembg submodule to path
rembg_path = Path(__file__).parent / "rembg"
if rembg_path.exists():
    sys.path.insert(0, str(rembg_path))

# Suppress warnings about unavailable CUDA provider (graceful fallback to CPU)
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import numpy as np
from PIL import Image
import cv2
from rembg import remove, new_session

try:
    import torch
except ImportError:  # torch is optional for the CPU-only preprocessing path
    torch = None

# Global sessions for performance (reused across calls)
_sessions = {}
_yolo_model = None


def get_session(model_name: str = "u2net"):
    """Get or create a rembg session for reuse with GPU support."""
    if model_name not in _sessions:
        # Disable checksum to allow locally cached/fine-tuned model files.
        from rembg.sessions.base import BaseSession

        original_checksum = BaseSession.checksum_disabled
        BaseSession.checksum_disabled = classmethod(lambda cls, *a, **kw: True)
        try:
            _sessions[model_name] = new_session(
                model_name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        finally:
            BaseSession.checksum_disabled = original_checksum
    return _sessions[model_name]


def get_yolo_model():
    """Get or create YOLO segmentation model for reuse."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        yolo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs/segment/checkpoints_yolo_seg/yolo_finetune/weights/best.pt"
        )
        _yolo_model = YOLO(yolo_path)
    return _yolo_model


def yolo_remove_background(image: Image.Image, return_mask: bool = False):
    """Remove background using YOLO segmentation model."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)
    model = get_yolo_model()
    results = model(img_np, verbose=False)

    h, w = img_np.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if results[0].masks is not None:
        for mask_data in results[0].masks.data:
            mask = mask_data.cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8))

    mask_img = Image.fromarray(combined_mask, mode="L")
    output = Image.merge("RGBA", (*image.split(), mask_img))

    if return_mask:
        return output, mask_img
    return output


def refine_background_mask(mask: Image.Image) -> Image.Image:
    """
    Improved mask refinement:
    1. Otsu threshold for clean binary separation
    2. Keep largest component, fit min-area rotated rectangle
    3. Create rounded-rectangle mask (tray shape)
    4. Intersect with original mask to preserve fine details
    """
    mask_np = np.array(mask.convert("L"))

    # 1. Otsu threshold for clean binary
    _, binary = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not np.any(binary):
        return mask

    # 2. Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        largest = binary

    h, w = largest.shape

    # 3. Find contours of largest component
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    main_contour = max(contours, key=cv2.contourArea)

    # 4. Fit min-area rotated rectangle + slight expansion (2%)
    rect = cv2.minAreaRect(main_contour)
    center, size, angle = rect
    size = (size[0] * 1.04, size[1] * 1.04)  # 4% expansion
    rect = (center, size, angle)

    # 5. Create rounded rectangle mask
    box_pts = cv2.boxPoints(rect)
    box_pts = np.int32(box_pts)

    # Draw rotated rectangle
    rect_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(rect_mask, [box_pts], -1, 255, -1)

    # Round corners with morphological closing + opening
    kernel = np.ones((15, 15), np.uint8)
    rect_rounded = cv2.morphologyEx(rect_mask, cv2.MORPH_CLOSE, kernel)
    rect_rounded = cv2.morphologyEx(rect_rounded, cv2.MORPH_OPEN, kernel)

    # 6. Use rectangle as base, intersect with original raw mask to keep tray details
    # but remove bottom-right artifacts that fall outside the rect
    refined = cv2.bitwise_and(largest, rect_rounded)

    # Fill any small holes in the result
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # Smooth edges
    blurred = cv2.GaussianBlur(refined.astype(np.float32), (9, 9), 0)
    _, smoothed = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    smoothed = smoothed.astype(np.uint8)

    return Image.fromarray(smoothed, mode="L")


def _odd_kernel_size(value: int, minimum: int = 3, maximum: int = 151) -> int:
    value = max(minimum, min(maximum, int(value)))
    return value if value % 2 == 1 else value + 1


def _dark_tray_candidate(image_np: np.ndarray) -> np.ndarray:
    """Approximate black tray pixels while excluding the white table."""
    red, green, blue = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)

    dark_neutral = (
        (max_channel < 125)
        & (min_channel < 110)
        & ((max_channel - min_channel) < 90)
    )
    # Luminance promotes uint8 to float64, so only evaluate it where the cheap
    # integer tests already passed instead of over the whole image.
    if dark_neutral.any():
        luminance = (
            0.299 * red[dark_neutral]
            + 0.587 * green[dark_neutral]
            + 0.114 * blue[dark_neutral]
        )
        dark_neutral[dark_neutral] = luminance < 120

    very_dark = max_channel < 70
    return dark_neutral | very_dark


def _dark_tray_retention(
    image_np: np.ndarray, mask_np: np.ndarray, dark: np.ndarray | None = None
) -> float | None:
    dark = _dark_tray_candidate(image_np) if dark is None else dark
    dark_count = int(dark.sum())
    if dark_count < max(500, int(dark.size * 0.002)):
        return None

    foreground = mask_np > 127
    return float((dark & foreground).sum() / dark_count)


def preserve_dark_tray_in_mask(
    image: Image.Image,
    mask: Image.Image,
    min_dark_retention: float = 0.4,
) -> Image.Image:
    """
    Repair rembg masks that keep meat but drop the black tray.

    The matching model expects tray + meat after table removal. Generic rembg can
    segment only the salient meat, so this adds the largest nearby black tray
    component back into the mask when dark tray retention is very low.
    """
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))
    foreground = mask_np > 127

    if not foreground.any():
        return mask

    dark = _dark_tray_candidate(image_np)
    retention = _dark_tray_retention(image_np, mask_np, dark)
    if retention is None or retention >= min_dark_retention:
        return mask

    h, w = mask_np.shape
    ys, xs = np.where(foreground)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    fg_w = right - left + 1
    fg_h = bottom - top + 1

    pad_x = max(80, int(fg_w * 0.55), int(w * 0.06))
    pad_y = max(80, int(fg_h * 0.55), int(h * 0.06))

    roi = np.zeros((h, w), dtype=bool)
    roi[
        max(0, top - pad_y) : min(h, bottom + pad_y + 1),
        max(0, left - pad_x) : min(w, right + pad_x + 1),
    ] = True

    tray_pixels = (dark & roi).astype(np.uint8) * 255
    if not np.any(tray_pixels):
        return mask

    close_size = _odd_kernel_size(max(31, int(max(fg_w, fg_h) * 0.04)), maximum=101)
    open_size = _odd_kernel_size(max(5, int(max(fg_w, fg_h) * 0.01)), maximum=21)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    tray_pixels = cv2.morphologyEx(tray_pixels, cv2.MORPH_CLOSE, close_kernel)
    tray_pixels = cv2.morphologyEx(tray_pixels, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        tray_pixels, connectivity=8
    )
    if num_labels <= 1:
        return mask

    fg_area = int(foreground.sum())
    min_component_area = max(1000, int(fg_area * 0.08))
    candidates = [
        label
        for label in range(1, num_labels)
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area
    ]
    if not candidates:
        return mask

    largest_label = max(candidates, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    largest_component = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask

    main_contour = max(contours, key=cv2.contourArea)
    tray_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(main_contour)
    cv2.drawContours(tray_mask, [hull], -1, 255, -1)

    tray_area = int((tray_mask > 0).sum())
    image_area = h * w
    fg_inside_tray = float((foreground & (tray_mask > 0)).sum() / fg_area)

    if fg_inside_tray < 0.5:
        return mask
    if tray_area < fg_area * 0.5 or tray_area > image_area * 0.6:
        return mask

    repaired = cv2.bitwise_or(mask_np, tray_mask)
    repaired = cv2.morphologyEx(
        repaired,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    return Image.fromarray(repaired.astype(np.uint8), mode="L")


def _blend_to_background(
    source_np: np.ndarray, bg: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """source*alpha + bg*(1-alpha) in float32, on the GPU when there is one.

    torch matches numpy here: IEEE float32 arithmetic and round-half-to-even,
    so the bytes are the same either way.
    """
    if torch is not None and torch.cuda.is_available():
        try:
            dev = torch.device("cuda")
            src_t = torch.from_numpy(source_np).to(dev, torch.float32)
            bg_t = torch.from_numpy(bg).to(dev, torch.float32)
            a_t = torch.from_numpy(alpha).to(dev, torch.float32)
            out = (src_t * a_t + bg_t * (1.0 - a_t)).round().to(torch.uint8)
            return out.cpu().numpy()
        except Exception:
            pass
    source_float = source_np.astype(np.float32)
    bg_float = bg.astype(np.float32)
    return (source_float * alpha + bg_float * (1.0 - alpha)).round().astype(np.uint8)


def apply_mask_to_rgb(
    image: Image.Image,
    mask: Image.Image,
    rembg_output: Image.Image | None = None,
    rembg_mask: Image.Image | None = None,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Apply alpha mask to original RGB while giving removed pixels stable RGB."""
    original_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L")).astype(np.float32)
    alpha = np.clip(mask_np / 255.0, 0.0, 1.0)[:, :, None]

    if rembg_output is not None:
        source_np = np.array(rembg_output.convert("RGBA"))[:, :, :3]
    else:
        source_np = original_np.copy()

    if rembg_mask is not None:
        rembg_mask_np = np.array(rembg_mask.convert("L"))
    elif rembg_output is not None and rembg_output.mode == "RGBA":
        rembg_mask_np = np.array(rembg_output.split()[3])
    else:
        rembg_mask_np = mask_np

    final_fg = mask_np > 10
    original_fg = rembg_mask_np > 10
    added_fg = final_fg & ~original_fg
    # _dark_tray_candidate is a full-image pass; nothing was added means nothing to classify.
    added_dark_tray = (
        added_fg & _dark_tray_candidate(original_np)
        if added_fg.any()
        else added_fg
    )

    source_np = source_np.copy()
    source_np[added_dark_tray] = original_np[added_dark_tray]

    # New non-tray pixels usually come from mask repair or geometric refinement;
    # keep those white so table marks do not reappear in the matcher input.
    bg = np.full_like(source_np, background_color, dtype=np.uint8)
    source_np[added_fg & ~added_dark_tray] = bg[added_fg & ~added_dark_tray]

    rgb = _blend_to_background(source_np, bg, alpha)
    return Image.merge("RGBA", (*Image.fromarray(rgb, mode="RGB").split(), mask))


def remove_background(
    image: Image.Image,
    use_session: bool = True,
    return_mask: bool = False,
    refine_mask: bool = True,
    model_name: str = "u2net",
    preserve_dark_tray: bool = True,
):
    """
    Remove background from image using rembg u2net model with GPU support.

    Args:
        image: PIL Image in any mode
        use_session: If True, reuse cached session for performance
        return_mask: If True, return tuple (image, mask). If False, return only image.
        refine_mask: If True, apply post-processing to keep only the largest component.
        preserve_dark_tray: If True, repair masks that remove the black tray.

    Returns:
        PIL Image with background removed (RGBA with transparency)
        Optional: If return_mask=True, returns (image, mask) tuple where mask is grayscale
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    if use_session:
        session = get_session(model_name)
        output = remove(image, session=session)
    else:
        output = remove(image, model_name=model_name)

    if refine_mask:
        # Extract mask
        if output.mode == "RGBA":
            mask = output.split()[3]
        else:
            mask = output.convert("RGBA").split()[3]

        # Refine mask
        refined_mask = refine_background_mask(mask)
        repaired_tray = False
        if preserve_dark_tray:
            before_repair = np.array(refined_mask.convert("L"), dtype=np.int16)
            refined_mask = preserve_dark_tray_in_mask(image, refined_mask)
            after_repair = np.array(refined_mask.convert("L"), dtype=np.int16)
            repaired_tray = bool(np.any(after_repair > before_repair + 10))

        # Keep the original rembg RGB path unless the tray repair actually added
        # pixels. This avoids perturbing already-correct matcher inputs.
        if repaired_tray:
            output = apply_mask_to_rgb(
                image,
                refined_mask,
                rembg_output=output,
                rembg_mask=mask,
            )
        else:
            if output.mode != "RGBA":
                output = output.convert("RGBA")
            rgb = output.split()[:3]
            output = Image.merge("RGBA", (*rgb, refined_mask))

    if return_mask:
        # Extract alpha channel as mask (grayscale)
        if output.mode == "RGBA":
            mask = output.split()[3]  # Get alpha channel
        else:
            # Convert to RGBA first if needed
            output_rgba = output.convert("RGBA")
            mask = output_rgba.split()[3]
        return output, mask

    return output


def _detect_top_horizontal_line(mask: Image.Image):
    """
    Detect the top-most near-horizontal line from a mask.

    Returns:
        Tuple (angle_deg, (x1, y1, x2, y2)) or (0.0, None) if not found.
    """
    mask_np = np.array(mask.convert("L"))

    # Use a cleaned binary mask for robust angle detection only.
    # The original mask remains unchanged for downstream rotation/cropping.
    blur = cv2.GaussianBlur(mask_np, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not np.any(binary):
        return 0.0, None

    # Remove fuzzy edges and fill small gaps before line detection.
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    edges = cv2.Canny(binary, 50, 150)
    h, w = binary.shape[:2]
    min_len = max(40, int(w * 0.25))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_len,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        lines = []

    candidates = []
    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        angle = ((angle + 90) % 180) - 90
        abs_angle = abs(angle)
        length = float(np.hypot(dx, dy))

        if abs_angle <= 30 and length >= min_len:
            y_mid = (y1 + y2) / 2.0
            candidates.append(
                (y_mid, -length, float(angle), (int(x1), int(y1), int(x2), int(y2)))
            )

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        top_y = candidates[0][0]
        top_band = [c for c in candidates if c[0] <= top_y + max(8, h * 0.02)]
        top_band.sort(key=lambda x: x[1])

        _, _, angle, line_coords = top_band[0]
        return angle, line_coords

    # Fallback: estimate top boundary line from mask profile.
    fg = binary > 0
    if not np.any(fg):
        return 0.0, None

    top_points_x = []
    top_points_y = []
    for x in range(w):
        ys = np.where(fg[:, x])[0]
        if ys.size > 0:
            top_points_x.append(x)
            top_points_y.append(int(ys[0]))

    if len(top_points_x) < max(10, int(w * 0.15)):
        return 0.0, None

    x_arr = np.array(top_points_x, dtype=np.float32)
    y_arr = np.array(top_points_y, dtype=np.float32)

    # Keep stable middle section to avoid noisy boundary ends.
    q_lo, q_hi = np.percentile(x_arr, [15, 85])
    keep = (x_arr >= q_lo) & (x_arr <= q_hi)
    x_fit = x_arr[keep]
    y_fit = y_arr[keep]

    if x_fit.size < 8:
        x_fit = x_arr
        y_fit = y_arr

    # Linear fit: y = m*x + b
    m, b = np.polyfit(x_fit, y_fit, 1)
    angle = float(np.degrees(np.arctan(m)))

    x1 = int(np.min(x_fit))
    x2 = int(np.max(x_fit))
    y1 = int(np.clip(m * x1 + b, 0, h - 1))
    y2 = int(np.clip(m * x2 + b, 0, h - 1))

    return angle, (x1, y1, x2, y2)


def detect_top_line_tilt_angle(mask: Image.Image) -> float:
    """
    Detect tilt angle from the most top near-horizontal line in a segmentation mask.

    Args:
        mask: Grayscale mask where foreground is high values

    Returns:
        Angle in degrees of the detected top line. 0.0 if no reliable line is found.
    """
    angle, _ = _detect_top_horizontal_line(mask)
    return float(angle)


def draw_detected_and_horizontal_line_on_mask(mask: Image.Image) -> Image.Image:
    """
    Draw the detected top line and ideal horizontal line on the mask.

    Colors:
        Red: detected top line
        Green: perfect horizontal line through the detected line midpoint

    Args:
        mask: Grayscale mask

    Returns:
        RGB PIL Image for visualization
    """
    mask_l = np.array(mask.convert("L"))
    vis = cv2.cvtColor(mask_l, cv2.COLOR_GRAY2RGB)

    angle, line_coords = _detect_top_horizontal_line(mask)
    h, w = mask_l.shape[:2]

    if line_coords is not None:
        x1, y1, x2, y2 = line_coords

        # Detected line (red)
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # Perfect horizontal reference line (green) at same vertical center
        y_ref = int(round((y1 + y2) / 2.0))
        y_ref = int(np.clip(y_ref, 0, h - 1))
        cv2.line(vis, (0, y_ref), (w - 1, y_ref), (0, 255, 0), 3)

        # Mark endpoints for readability
        cv2.circle(vis, (x1, y1), 6, (255, 255, 0), -1)
        cv2.circle(vis, (x2, y2), 6, (255, 255, 0), -1)

        cv2.putText(
            vis,
            f"angle={angle:.2f} deg",
            (10, min(h - 10, y_ref + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            "No top line detected",
            (10, min(h - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(vis, mode="RGB")


def rotate_image_and_mask(image: Image.Image, mask: Image.Image, angle: float):
    """
    Rotate image and mask by the same angle to keep them spatially aligned.

    Args:
        image: PIL Image (RGB or RGBA)
        mask: PIL grayscale mask
        angle: Rotation angle in degrees (OpenCV convention)

    Returns:
        Tuple of (rotated_image, rotated_mask)
    """
    if abs(angle) < 1e-3:
        return image, mask

    image_np = np.array(image)
    mask_np = np.array(mask.convert("L"))
    h, w = mask_np.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if image.mode == "RGBA":
        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3]

        rotated_rgb = cv2.warpAffine(
            rgb,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        rotated_alpha = cv2.warpAffine(
            alpha,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated_image = Image.fromarray(
            np.dstack([rotated_rgb, rotated_alpha]).astype(np.uint8), mode="RGBA"
        )
    else:
        rotated_rgb = cv2.warpAffine(
            image_np,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        rotated_image = Image.fromarray(rotated_rgb.astype(np.uint8), mode="RGB")

    rotated_mask = cv2.warpAffine(
        mask_np,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, Image.fromarray(rotated_mask.astype(np.uint8), mode="L")


def crop_image_by_mask(image: Image.Image, mask: Image.Image, padding: int = 0):
    """
    Crop image and mask together using foreground bounds from mask.

    Args:
        image: PIL Image to crop
        mask: Grayscale mask used for bounds
        padding: Optional extra pixels around content

    Returns:
        Tuple of (cropped_image, cropped_mask)
    """
    mask_np = np.array(mask.convert("L"))
    fg = mask_np > 10

    if not fg.any():
        return image, mask

    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return image, mask

    top = max(0, int(row_indices[0]) - padding)
    bottom = min(image.height, int(row_indices[-1]) + padding + 1)
    left = max(0, int(col_indices[0]) - padding)
    right = min(image.width, int(col_indices[-1]) + padding + 1)

    box = (left, top, right, bottom)
    return image.crop(box), mask.crop(box)


def fill_mask_holes(mask: Image.Image) -> Image.Image:
    """Fill enclosed background holes in a foreground mask."""
    mask_np = np.array(mask.convert("L"))
    _, foreground = cv2.threshold(mask_np, 10, 255, cv2.THRESH_BINARY)
    background = cv2.bitwise_not(foreground)
    border_bg = np.zeros_like(background)
    border_bg[0, :] = background[0, :]
    border_bg[-1, :] = background[-1, :]
    border_bg[:, 0] = background[:, 0]
    border_bg[:, -1] = background[:, -1]

    _, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    outside_labels = set(np.unique(labels[border_bg > 0]))
    filled = foreground.copy()
    for label in range(1, len(stats)):
        if label not in outside_labels:
            filled[labels == label] = 255
    return Image.fromarray(filled, mode="L")


def correct_horizontal_balance(image: Image.Image) -> Image.Image:
    """
    Rotate image to correct horizontal tilt/balance.
    Detects the predominant horizontal edges and rotates to make them level.

    Args:
        image: PIL Image (RGBA or RGB)

    Returns:
        Rotated PIL Image
    """
    # Convert to numpy array
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 100, 200)

    # Use Hough line transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) == 0:
        # No lines detected, return original
        return image

    # Extract angles from near-horizontal lines (those close to 0 or 180 degrees)
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert theta to degrees
        theta_deg = np.degrees(theta)

        # Look for lines close to horizontal (near 0 or 180 degrees)
        # Accept lines within ±15 degrees of horizontal
        if theta_deg < 15 or theta_deg > 165:
            if theta_deg > 90:
                theta_deg = theta_deg - 180
            angles.append(theta_deg)

    if not angles:
        return image

    # Calculate median angle (most robust to outliers)
    rotation_angle = np.median(angles)

    # Apply rotation to correct tilt
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Apply rotation to all image modes
    if image.mode == "RGBA":
        # Handle RGBA separately to preserve alpha
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_array = np.array(image)
        img_alpha = img_array[:, :, 3]

        rotated_bgr = cv2.warpAffine(img_rgb, M, (w, h), borderValue=(255, 255, 255))
        rotated_alpha = cv2.warpAffine(img_alpha, M, (w, h), borderValue=0)

        rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
        rotated_rgba = np.dstack([rotated_rgb, rotated_alpha])
        return Image.fromarray(rotated_rgba, mode="RGBA")
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        rotated_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderValue=(255, 255, 255))
        rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rotated_rgb, mode="RGB")


def smart_crop(image: Image.Image, padding: int = 0) -> Image.Image:
    """
    Crop image to remove excess white/transparent space.
    Finds bounding box of non-background content with optional padding.

    Args:
        image: PIL Image (RGB or RGBA)
        padding: Pixels to keep around the content (default: 0 for tight crop)

    Returns:
        Cropped PIL Image
    """
    if image.mode == "RGBA":
        # Use alpha channel to find content bounds
        img_array = np.array(image)
        alpha = img_array[:, :, 3]

        # Find non-transparent pixels
        non_transparent = alpha > 10  # Use threshold instead of > 0

        if not non_transparent.any():
            return image

        rows = np.any(non_transparent, axis=1)
        cols = np.any(non_transparent, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return image

        top = max(0, row_indices[0] - padding)
        bottom = min(image.height, row_indices[-1] + padding + 1)
        left = max(0, col_indices[0] - padding)
        right = min(image.width, col_indices[-1] + padding + 1)

        return image.crop((left, top, right, bottom))
    else:
        # For RGB images, find non-white content
        img_array = np.array(image)

        # Consider pixels non-white if they're not close to (255, 255, 255)
        non_white = np.any(img_array < 240, axis=2)

        if not non_white.any():
            return image

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return image

        top = max(0, row_indices[0] - padding)
        bottom = min(image.height, row_indices[-1] + padding + 1)
        left = max(0, col_indices[0] - padding)
        right = min(image.width, col_indices[-1] + padding + 1)

        return image.crop((left, top, right, bottom))


# u2net + the cv2 morphology in remove_background scale with input pixels; 4 is
# what the matching gallery was built with. Set to 1 to segment at full res.
MATCHING_DOWNSCALE = int(os.getenv("MATCHING_DOWNSCALE", "4"))


def _rotate_mask(mask: Image.Image, angle: float) -> Image.Image:
    """Mask half of rotate_image_and_mask, pixel-for-pixel."""
    if abs(angle) < 1e-3:
        return mask
    mask_np = np.array(mask.convert("L"))
    h, w = mask_np.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        mask_np, matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return Image.fromarray(rotated.astype(np.uint8), mode="L")


def _pick_tilt_angle(mask: Image.Image) -> float:
    """Angle whose rotation leaves the least residual tilt.

    Rotates only the mask: the probe reads nothing else, so rotating the RGBA
    image for the losing angle is wasted work at full resolution.
    """
    angle = detect_top_line_tilt_angle(mask)
    residual_pos = abs(detect_top_line_tilt_angle(_rotate_mask(mask, angle)))
    residual_neg = abs(detect_top_line_tilt_angle(_rotate_mask(mask, -angle)))
    return angle if residual_pos <= residual_neg else -angle


def preprocess_for_matching_with_rgba(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    if MATCHING_DOWNSCALE > 1:
        image = image.resize((
            max(1, image.width // MATCHING_DOWNSCALE),
            max(1, image.height // MATCHING_DOWNSCALE),
        ))
    proc_img, bg_mask = remove_background(image, return_mask=True, model_name="u2net")
    proc_img, bg_mask = rotate_image_and_mask(proc_img, bg_mask, _pick_tilt_angle(bg_mask))

    proc_img, bg_mask = crop_image_by_mask(proc_img, bg_mask, padding=0)
    proc_img, bg_mask = crop_image_by_mask(proc_img, bg_mask, padding=0)
    bg_mask = fill_mask_holes(bg_mask)
    proc_img = apply_mask_to_rgb(proc_img, bg_mask, rembg_output=proc_img)
    rgb_image = proc_img.convert("RGB")
    rgba_image = proc_img.convert("RGBA")
    return rgb_image, rgba_image


def preprocess_for_matching(image: Image.Image) -> Image.Image:
    return preprocess_for_matching_with_rgba(image)[0]


def preprocess_for_anomaly(image: Image.Image) -> Image.Image:
    proc_img, bg_mask = remove_background(image, return_mask=True, model_name="u2net")
    detected_angle = detect_top_line_tilt_angle(bg_mask)

    img_pos, mask_pos = rotate_image_and_mask(proc_img, bg_mask, detected_angle)
    residual_pos = abs(detect_top_line_tilt_angle(mask_pos))

    img_neg, mask_neg = rotate_image_and_mask(proc_img, bg_mask, -detected_angle)
    residual_neg = abs(detect_top_line_tilt_angle(mask_neg))

    if residual_pos <= residual_neg:
        proc_img, bg_mask = img_pos, mask_pos
    else:
        proc_img, bg_mask = img_neg, mask_neg

    proc_img, bg_mask = crop_image_by_mask(proc_img, bg_mask, padding=0)
    proc_img, _ = crop_image_by_mask(proc_img, bg_mask, padding=0)
    return proc_img.convert("RGB")


def preprocess(image: Image.Image) -> Image.Image:
    return preprocess_for_anomaly(image)
