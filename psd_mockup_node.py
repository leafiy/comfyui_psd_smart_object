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
        if not layers:
            return []
        return [layers[0]]
    lookup = {_normalize_name(name) for name in names if name.strip()}
    if not lookup:
        return layers
    result = [layer for layer in layers if _normalize_name(layer.name) in lookup]
    if not result:
        raise ValueError(f"No smart-object layers matched {names}.")
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


class PSDFileUploader:
    @classmethod
    def INPUT_TYPES(cls):
        psd_files = _list_input_files(ALLOWED_PSD_EXTENSIONS)
        if not psd_files:
            psd_files = ["<upload or pick PSD>"]
        return {
            "required": {
                "psd_file": (
                    psd_files,
                    {
                        "file_upload": True,
                        "file_upload_types": ["psd", "psb"],
                        "tooltip": "Click the upload icon to add a PSD/PSB into ComfyUI/input",
                    },
                )
            }
        }

    CATEGORY = "psd/mockup"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("psd_path",)
    FUNCTION = "upload"

    def upload(self, psd_file: str):
        """Return the resolved path inside ComfyUI/input for the chosen PSD."""
        path = _resolve_input_path(psd_file, ALLOWED_PSD_EXTENSIONS)
        return (path,)


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
        psd = PSDImage.open(psd_path)
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
        psd_files = _list_input_files(ALLOWED_PSD_EXTENSIONS)
        if not psd_files:
            psd_files = ["<upload PSD via ⬆ icon>"]
        return {
            "required": {
                "psd_file": (psd_files,),
                "mockup_image": ("IMAGE",),
            },
            "optional": {
                "smart_object_names": ("STRING", {"default": "", "multiline": False}),
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
    ):
        resolved_psd = _resolve_input_path(psd_file, ALLOWED_PSD_EXTENSIONS)
        psd = PSDImage.open(resolved_psd)
        source_image = _tensor_to_pil(mockup_image)
        if source_image is None:
            raise ValueError("mockup_image input is required. Use MockupImageUpload or any IMAGE source.")

        layers = _collect_smart_layers(psd)
        selected = _select_layers(layers, _split_names(smart_object_names))
        if not selected:
            raise ValueError("No smart-object layers available inside the PSD.")

        canvas_size = psd.size
        base = _compose_without_layers(psd, [layer.layer_id for layer in selected])
        base_rgba = base.convert("RGBA")

        metadata = []
        for layer in selected:
            quad = layer.transform_box
            if quad is None:
                left, top, right, bottom = layer.bbox
                quad = (left, top, right, top, right, bottom, left, bottom)
            warped = _warp_image_to_canvas(source_image, quad, canvas_size)
            mask = _polygon_mask(quad, canvas_size)
            base_rgba = Image.composite(warped, base_rgba, mask)
            metadata.append(
                {
                    "layer_id": layer.layer_id,
                    "name": layer.name,
                    "quad": quad,
                }
            )

        result_tensor = _pil_to_tensor(base_rgba)
        saved_path = _save_png(base_rgba, resolved_psd)
        debug_payload = {
            "psd": os.path.basename(resolved_psd),
            "layer_count": len(selected),
            "layers": metadata,
        }
        return (result_tensor, json.dumps(debug_payload, indent=2), saved_path)


NODE_CLASS_MAPPINGS = {
    "PSDFileUploader": PSDFileUploader,
    "PSDSmartObjectInspector": PSDSmartObjectInspector,
    "PSDMockupEmbedder": PSDMockupEmbedder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDFileUploader": "PSD File Upload",
    "PSDSmartObjectInspector": "PSD → Smart Object Info",
    "PSDMockupEmbedder": "PSD Mockup Embedder",
}
