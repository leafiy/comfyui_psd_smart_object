"""
ComfyUI custom nodes for embedding images into PSD smart object mockups.

The main node takes a PSD mockup file together with an input image, finds
smart-object layers, and replaces their content by warping the input image
to the smart-object's perspective box.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import Layer, SmartObjectLayer

try:
    import torch
except Exception:  # pragma: no cover - torch is always available inside ComfyUI
    torch = None

try:
    import folder_paths
except Exception:  # pragma: no cover - allows running outside ComfyUI
    folder_paths = None


ALLOWED_PSD_EXTENSIONS = (".psd", ".psb")
QUAD_INDICES = ((0, 1), (2, 3), (4, 5), (6, 7))
PNG_SUFFIX = ".png"


@dataclass
class SmartLayerInfo:
    """Small helper that keeps metadata for a smart-object layer."""

    layer_id: int
    name: str
    visible: bool
    bbox: Tuple[int, int, int, int]
    transform_box: Optional[Tuple[float, ...]]

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "name": self.name,
            "visible": self.visible,
            "bbox": self.bbox,
            "transform_box": self.transform_box,
        }


def _resolve_input_path(filename: str, allowed_exts: Sequence[str]) -> str:
    if not filename:
        raise ValueError("A file name must be provided.")
    if os.path.isabs(filename):
        candidate = filename
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"File not found: {candidate}")
    else:
        search_roots: List[str] = []
        if folder_paths:
            getter = getattr(folder_paths, "get_input_directory", None)
            if callable(getter):
                search_roots.append(getter())
            get_extra = getattr(folder_paths, "get_input_directories", None)
            if callable(get_extra):
                search_roots.extend(get_extra())
        search_roots.append(os.getcwd())
        candidate = ""
        for root in search_roots:
            path = os.path.join(root, filename)
            if os.path.exists(path):
                candidate = path
                break
        if not candidate:
            raise FileNotFoundError(
                f"Unable to resolve '{filename}'. Place the file under ComfyUI/input or provide an absolute path."
            )
    _, ext = os.path.splitext(candidate)
    ext = ext.lower()
    if allowed_exts and ext not in allowed_exts:
        raise ValueError(f"Unsupported file extension '{ext}'. Expected one of {allowed_exts}.")
    return candidate


def _list_input_files(allowed_exts: Sequence[str]) -> List[str]:
    if not folder_paths or not hasattr(folder_paths, "get_filename_list"):
        return []
    try:
        files = folder_paths.get_filename_list("input")
    except Exception:
        return []
    if not allowed_exts:
        return files
    allowed = tuple(ext.lower() for ext in allowed_exts)
    filtered = [name for name in files if name.lower().endswith(allowed)]
    return filtered or files


def _collect_smart_layers(group: Iterable[Layer]) -> List[SmartLayerInfo]:
    layers: List[SmartLayerInfo] = []

    def walk(sub_layers: Iterable[Layer]):
        for layer in sub_layers:
            if isinstance(layer, SmartObjectLayer):
                layers.append(
                    SmartLayerInfo(
                        layer_id=layer.layer_id,
                        name=layer.name,
                        visible=layer.is_visible(),
                        bbox=tuple(map(int, layer.bbox)),
                        transform_box=layer.smart_object.transform_box,
                    )
                )
            if layer.is_group():
                walk(layer)

    walk(group)
    return layers


def _tensor_to_pil(image_tensor) -> Image.Image:
    if image_tensor is None:
        return None
    if torch is None:
        raise RuntimeError("torch is required to convert IMAGE values. Are you running inside ComfyUI?")
    tensor = image_tensor
    if isinstance(tensor, list):
        tensor = tensor[0]
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    array = np.clip(tensor.numpy(), 0.0, 1.0)
    if array.ndim == 4:
        array = array[0]
    array = (array * 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array)


def _pil_to_tensor(image: Image.Image):
    if torch is None:
        raise RuntimeError("torch is required to convert PIL images back to ComfyUI tensors.")
    rgb = image.convert("RGB")
    array = np.array(rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, ...]
    return tensor


def _load_image_from_path(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path)
    return image.convert("RGBA")


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _select_layers(layers: List[SmartLayerInfo], names: Sequence[str]) -> List[SmartLayerInfo]:
    if not names or len(names) == 0:
        return layers[:1] if layers else []
    lookup = {_normalize_name(name) for name in names if name.strip()}
    if not lookup:
        return layers[:1] if layers else []
    result = [layer for layer in layers if _normalize_name(layer.name) in lookup]
    if not result:
        return layers[:1] if layers else []
    return result


def _split_names(raw_names: str) -> List[str]:
    if not raw_names:
        return []
    return [part.strip() for part in raw_names.split(",") if part.strip()]


def _warp_image_to_canvas(image: Image.Image, quad: Sequence[float], canvas_size: Tuple[int, int]) -> Image.Image:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - cv2 is installed via requirements
        raise RuntimeError("opencv-python-headless is required for perspective warping.") from exc

    if quad is None or len(quad) != 8:
        raise ValueError("transform_box must contain eight values.")
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("Canvas size must be positive.")

    image_rgba = image.convert("RGBA")
    src_w, src_h = image_rgba.size
    src_pts = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
    dst_pts = np.float32([[quad[i], quad[i + 1]] for i, _ in QUAD_INDICES])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    canvas_w, canvas_h = int(math.ceil(width)), int(math.ceil(height))
    rgba_array = np.array(image_rgba)
    warped = cv2.warpPerspective(
        rgba_array,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA")


def _compose_without_layers(psd: PSDImage, skip_ids: Sequence[int]) -> Image.Image:
    skip_set = set(skip_ids)

    def layer_filter(layer: Layer) -> bool:
        if hasattr(layer, "layer_id") and layer.layer_id in skip_set:
            return False
        return layer.is_visible()

    return psd.composite(layer_filter=layer_filter)


def _polygon_mask(quad: Sequence[float], canvas_size: Tuple[int, int]) -> Image.Image:
    mask = Image.new("L", tuple(map(int, canvas_size)), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(quad[i], quad[i + 1]) for i, _ in QUAD_INDICES], fill=255)
    return mask


def _build_layer_lookup(psd: PSDImage) -> dict:
    lookup = {}
    for layer in psd.descendants():
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is not None:
            lookup[layer_id] = layer
    return lookup


def _collect_clipping_stack(layer: Layer) -> List[Layer]:
    stack = [layer]
    parent = layer.parent
    if parent is None:
        return stack
    siblings = list(parent)
    try:
        idx = siblings.index(layer)
    except ValueError:
        return stack
    for sib in siblings[idx + 1 :]:
        if not getattr(sib, "clipping", False):
            break
        stack.append(sib)
    return stack


def _composite_layers(psd: PSDImage, layer_ids: Sequence[int]) -> Image.Image:
    include = set(layer_ids)

    def filter_fn(layer: Layer) -> bool:
        if hasattr(layer, "layer_id") and layer.layer_id in include:
            return True
        return False

    return psd.composite(layer_filter=filter_fn)


def _compute_shading_components(stack_image: Image.Image, base_image: Image.Image):
    stack_rgba = stack_image.convert("RGBA")
    base_rgba = base_image.convert("RGBA")
    stack_arr = np.array(stack_rgba).astype(np.float32) / 255.0
    base_arr = np.array(base_rgba).astype(np.float32) / 255.0
    base_rgb = base_arr[..., :3]
    stack_rgb = stack_arr[..., :3]
    base_alpha = base_arr[..., 3:4]

    multiplier = np.ones_like(base_rgb, dtype=np.float32)
    mask = base_alpha > 1e-3
    if np.any(mask):
        denom = np.clip(base_rgb, 1e-3, None)
        ratio = stack_rgb / denom
        ratio = np.clip(ratio, 0.0, 4.0)
        multiplier[mask.repeat(3, axis=2)] = ratio[mask.repeat(3, axis=2)]

    base_alpha_rgb = np.repeat(base_alpha, 3, axis=2)
    multiplier = multiplier * base_alpha_rgb + (1.0 - base_alpha_rgb)
    alpha = stack_arr[..., 3:4]
    alpha_img = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8).squeeze(-1), mode="L")
    return multiplier, alpha_img


def _apply_shading_map(image: Image.Image, multiplier: np.ndarray) -> Image.Image:
    if multiplier is None:
        return image
    rgba = image.convert("RGBA")
    arr = np.array(rgba).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    if multiplier.shape[:2] != (h, w):
        return image
    arr[..., :3] = np.clip(arr[..., :3] * multiplier, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGBA")


def _output_directory() -> str:
    if folder_paths and hasattr(folder_paths, "get_output_directory"):
        return folder_paths.get_output_directory()
    default_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def _save_png(image: Image.Image, psd_path: str) -> str:
    base_name = Path(psd_path).stem or "psd_mockup"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}{PNG_SUFFIX}"
    out_dir = _output_directory()
    out_path = os.path.join(out_dir, filename)
    image.save(out_path, "PNG")
    return out_path


class PSDSmartObjectInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_file": ("STRING", {"default": "", "multiline": False}),
                "include_hidden": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "psd/mockup"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"

    def inspect(self, psd_file: str, include_hidden: bool):
        psd_path = _resolve_input_path(psd_file, ALLOWED_PSD_EXTENSIONS)
        psd = PSDImage.open(psd_path, strict=False)
        layers = _collect_smart_layers(psd)
        payload = []
        for layer in layers:
            if not include_hidden and not layer.visible:
                continue
            payload.append(layer.to_dict())
        return (json.dumps({"psd": os.path.basename(psd_path), "layers": payload}, indent=2),)


class PSDMockupEmbedder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Upload or paste PSD/PSB path",
                        "tooltip": "Use the built-in upload button or paste a path manually",
                    },
                ),
                "mockup_image": ("IMAGE",),
            },
            "optional": {
                "smart_object_names": ("STRING", {"default": "", "multiline": False}),
                "output_width": (
                    "INT",
                    {
                        "default": 800,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Resize final image to this width (0 keeps original size)",
                    },
                ),
            },
        }

    CATEGORY = "psd/mockup"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("mockup_image", "debug_info", "png_path")
    FUNCTION = "apply"

    def apply(
        self,
        psd_file: str,
        mockup_image,
        smart_object_names: str = "",
        output_width: int = 800,
    ):
        resolved_psd = _resolve_input_path(psd_file, ALLOWED_PSD_EXTENSIONS)
        psd = PSDImage.open(resolved_psd, strict=False)
        source_image = _tensor_to_pil(mockup_image)
        if source_image is None:
            raise ValueError("mockup_image input is required. Use MockupImageUpload or any IMAGE source.")

        layers = _collect_smart_layers(psd)
        requested_names = _split_names(smart_object_names)
        selected = _select_layers(layers, requested_names)
        if not selected:
            raise ValueError("No smart-object layers available inside the PSD.")
        fallback_used = False
        if requested_names:
            requested_norm = {_normalize_name(name) for name in requested_names}
            selected_norm = {_normalize_name(layer.name) for layer in selected}
            fallback_used = not bool(requested_norm & selected_norm)

        layer_lookup = _build_layer_lookup(psd)
        canvas_size = psd.size
        base = _compose_without_layers(psd, [info.layer_id for info in selected])
        base_rgba = base.convert("RGBA")

        metadata = []
        for info in selected:
            layer = layer_lookup.get(info.layer_id)
            if layer is None:
                continue
            quad = info.transform_box
            if quad is None:
                left, top, right, bottom = info.bbox
                quad = (left, top, right, top, right, bottom, left, bottom)

            stack_layers = _collect_clipping_stack(layer)
            stack_ids = [l.layer_id for l in stack_layers if hasattr(l, "layer_id")]
            stack_image = _composite_layers(psd, stack_ids) if stack_ids else None
            base_image = _composite_layers(psd, [layer.layer_id])
            shading_multiplier = None
            alpha_mask = None
            if stack_image is not None:
                shading_multiplier, alpha_mask = _compute_shading_components(stack_image, base_image)

            warped = _warp_image_to_canvas(source_image, quad, canvas_size)
            warped = _apply_shading_map(warped, shading_multiplier)
            fallback_mask = _polygon_mask(quad, canvas_size)
            composite_mask = alpha_mask if alpha_mask and alpha_mask.getbbox() else fallback_mask
            base_rgba = Image.composite(warped, base_rgba, composite_mask)
            metadata.append(
                {
                    "layer_id": info.layer_id,
                    "name": info.name,
                    "quad": quad,
                }
            )

        if isinstance(output_width, (int, float)) and output_width and output_width > 0:
            target_w = int(output_width)
            if target_w > 0 and target_w != base_rgba.width:
                ratio = target_w / base_rgba.width
                target_h = max(1, int(round(base_rgba.height * ratio)))
                base_rgba = base_rgba.resize((target_w, target_h), Image.LANCZOS)

        result_tensor = _pil_to_tensor(base_rgba)
        saved_path = _save_png(base_rgba, resolved_psd)
        debug_payload = {
            "psd": os.path.basename(resolved_psd),
            "layer_count": len(selected),
            "layers": metadata,
            "output_size": {"width": base_rgba.width, "height": base_rgba.height},
        }
        if fallback_used:
            debug_payload["note"] = "Requested smart object names were not found; falling back to the first smart object layer."
        return (result_tensor, json.dumps(debug_payload, indent=2), saved_path)


NODE_CLASS_MAPPINGS = {
    "PSDSmartObjectInspector": PSDSmartObjectInspector,
    "PSDMockupEmbedder": PSDMockupEmbedder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDSmartObjectInspector": "PSD → Smart Object Info",
    "PSDMockupEmbedder": "PSD Mockup Embedder",
}
