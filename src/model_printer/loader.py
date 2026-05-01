"""Checkpoint loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
from typing import Any
from urllib.parse import unquote, urlparse


COMMON_STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "module",
    "ema",
    "teacher",
    "student",
)

HF_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".npz",
    ".bin",
    ".pth",
    ".pt",
)

SAFETENSORS_HEADER_LIMIT = 100 * 1024 * 1024


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be converted to a state_dict."""


@dataclass(frozen=True)
class TensorInfo:
    """Small serializable description of a tensor-like object."""

    shape: tuple[int, ...]
    dtype: str | None = None

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


StateDictInfo = dict[str, TensorInfo]


@dataclass(frozen=True)
class HuggingFaceReference:
    repo_id: str
    revision: str | None = None
    filename: str | None = None


def load_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    unsafe_load: bool = False,
    strip_module_prefix: bool = True,
    strip_prefixes: tuple[str, ...] = (),
) -> StateDictInfo:
    """Load a checkpoint reference and return a normalized state_dict summary.

    The returned dictionary contains only tensor metadata. Keeping metadata
    instead of tensors avoids carrying a large model in memory through the rest
    of the rendering pipeline.
    """

    checkpoint_ref = str(checkpoint_path)
    if is_huggingface_url(checkpoint_ref):
        normalized = load_huggingface_state_dict(
            checkpoint_ref,
            map_location=map_location,
            unsafe_load=unsafe_load,
        )
    else:
        normalized = load_local_state_dict(
            Path(checkpoint_path),
            map_location=map_location,
            unsafe_load=unsafe_load,
        )

    if strip_module_prefix:
        normalized = strip_common_prefix(normalized, "module.")

    for prefix in strip_prefixes:
        normalized = strip_common_prefix(normalized, prefix)

    return normalized


def load_local_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    unsafe_load: bool = False,
) -> StateDictInfo:
    """Load tensor metadata from a local checkpoint file."""

    path = Path(checkpoint_path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return load_npz_state_dict(path)
    if suffix == ".safetensors":
        return load_safetensors_state_dict(path)

    checkpoint = _torch_load(
        path,
        map_location=map_location,
        unsafe_load=unsafe_load,
    )
    state_dict = find_state_dict(checkpoint)
    return summarize_state_dict(state_dict)


def load_npz_state_dict(checkpoint_path: str | Path) -> StateDictInfo:
    """Load tensor metadata from a NumPy .npz archive."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointLoadError(f"Checkpoint not found: {path}")

    try:
        import numpy as np
    except ImportError as exc:
        raise CheckpointLoadError(
            "NumPy is required to read .npz files. Install it with "
            "`python -m pip install numpy`."
        ) from exc

    try:
        with np.load(path, allow_pickle=False) as archive:
            state_dict = {
                normalize_npz_key(key): TensorInfo(
                    shape=tuple(int(dim) for dim in archive[key].shape),
                    dtype=str(archive[key].dtype),
                )
                for key in archive.files
            }
    except ValueError as exc:
        raise CheckpointLoadError(
            "Failed to load .npz safely. Object arrays and pickled values are "
            "not supported; save numeric arrays instead."
        ) from exc
    except OSError as exc:
        raise CheckpointLoadError(f"Failed to load .npz checkpoint: {exc}") from exc

    if not state_dict:
        raise CheckpointLoadError("The .npz archive contains no arrays.")

    return dict(sorted(state_dict.items()))


def load_safetensors_state_dict(checkpoint_path: str | Path) -> StateDictInfo:
    """Read tensor metadata from a local safetensors file without loading data."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointLoadError(f"Checkpoint not found: {path}")

    try:
        with path.open("rb") as file:
            header_length_bytes = file.read(8)
            if len(header_length_bytes) != 8:
                raise CheckpointLoadError(f"Invalid safetensors file: {path}")
            header_length = struct.unpack("<Q", header_length_bytes)[0]
            if header_length > SAFETENSORS_HEADER_LIMIT:
                raise CheckpointLoadError(
                    f"Safetensors header is too large: {header_length} bytes"
                )
            header = json.loads(file.read(header_length).decode("utf-8"))
    except OSError as exc:
        raise CheckpointLoadError(f"Failed to read safetensors file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointLoadError(f"Invalid safetensors metadata: {exc}") from exc

    state_dict: StateDictInfo = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, Mapping):
            continue
        shape = value.get("shape")
        if shape is None:
            continue
        state_dict[str(key)] = TensorInfo(
            shape=tuple(int(dim) for dim in shape),
            dtype=str(value.get("dtype")) if value.get("dtype") else None,
        )

    if not state_dict:
        raise CheckpointLoadError("The safetensors file contains no tensors.")

    return dict(sorted(state_dict.items()))


def normalize_npz_key(key: str) -> str:
    """Normalize common NPZ path separators to state_dict-style dotted keys."""

    return key.replace("/", ".").strip(".")


def load_huggingface_state_dict(
    model_url: str,
    *,
    map_location: str = "cpu",
    unsafe_load: bool = False,
) -> StateDictInfo:
    """Load tensor metadata from a Hugging Face model URL."""

    reference = parse_huggingface_reference(model_url)
    api, hf_hub_download = _huggingface_client()

    if reference.filename and reference.filename.endswith(".safetensors"):
        return load_huggingface_safetensors_metadata(api, reference)

    if reference.filename is None:
        try:
            return load_huggingface_safetensors_metadata(api, reference)
        except CheckpointLoadError:
            pass

    filenames = resolve_huggingface_weight_files(api, reference)
    if not filenames:
        raise CheckpointLoadError(
            f"No supported weight files found in Hugging Face repo "
            f"{reference.repo_id!r}."
        )

    state_dict: StateDictInfo = {}
    for filename in filenames:
        try:
            downloaded = hf_hub_download(
                repo_id=reference.repo_id,
                filename=filename,
                revision=reference.revision,
            )
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to download {filename!r} from {reference.repo_id!r}: {exc}"
            ) from exc

        state_dict.update(
            load_local_state_dict(
                downloaded,
                map_location=map_location,
                unsafe_load=unsafe_load,
            )
        )

    return dict(sorted(state_dict.items()))


def load_huggingface_safetensors_metadata(
    api: Any,
    reference: HuggingFaceReference,
) -> StateDictInfo:
    """Read safetensors tensor metadata through huggingface_hub APIs."""

    try:
        if reference.filename:
            metadata = api.parse_safetensors_file_metadata(
                repo_id=reference.repo_id,
                filename=reference.filename,
                repo_type="model",
                revision=reference.revision,
            )
            files_metadata = {reference.filename: metadata}
        else:
            repo_metadata = api.get_safetensors_metadata(
                repo_id=reference.repo_id,
                repo_type="model",
                revision=reference.revision,
            )
            files_metadata = repo_metadata.files_metadata
    except Exception as exc:
        raise CheckpointLoadError(
            f"Could not read safetensors metadata from Hugging Face: {exc}"
        ) from exc

    state_dict = summarize_huggingface_safetensors_metadata(files_metadata)
    if not state_dict:
        raise CheckpointLoadError("The Hugging Face safetensors metadata is empty.")
    return state_dict


def summarize_huggingface_safetensors_metadata(
    files_metadata: Any,
) -> StateDictInfo:
    """Convert huggingface_hub SafetensorsFileMetadata objects to TensorInfo."""

    state_dict: StateDictInfo = {}
    if isinstance(files_metadata, Mapping):
        metadata_items = files_metadata.values()
    else:
        metadata_items = files_metadata

    for file_metadata in metadata_items:
        tensors = getattr(file_metadata, "tensors", {})
        for key, tensor in tensors.items():
            shape = getattr(tensor, "shape", None)
            dtype = getattr(tensor, "dtype", None)
            if shape is None and isinstance(tensor, Mapping):
                shape = tensor.get("shape")
                dtype = tensor.get("dtype")
            if shape is None:
                continue
            state_dict[str(key)] = TensorInfo(
                shape=tuple(int(dim) for dim in shape),
                dtype=str(dtype) if dtype else None,
            )
    return dict(sorted(state_dict.items()))


def resolve_huggingface_weight_files(
    api: Any,
    reference: HuggingFaceReference,
) -> list[str]:
    """Resolve the weight file or files to inspect for a Hugging Face URL."""

    if reference.filename:
        return [reference.filename]

    try:
        info = api.model_info(
            reference.repo_id,
            revision=reference.revision,
            files_metadata=False,
        )
    except Exception as exc:
        raise CheckpointLoadError(
            f"Could not inspect Hugging Face repo {reference.repo_id!r}: {exc}"
        ) from exc

    filenames = sorted(
        sibling.rfilename
        for sibling in getattr(info, "siblings", [])
        if getattr(sibling, "rfilename", None)
    )
    return select_huggingface_weight_files(filenames)


def select_huggingface_weight_files(filenames: list[str]) -> list[str]:
    """Pick the best supported checkpoint files from a Hugging Face repo."""

    filename_set = set(filenames)
    if "model.safetensors" in filename_set:
        return ["model.safetensors"]

    sharded_safetensors = [
        name
        for name in filenames
        if re.search(r"-\d{5}-of-\d{5}\.safetensors$", name)
    ]
    if sharded_safetensors:
        return sorted(sharded_safetensors)

    if "pytorch_model.bin" in filename_set:
        return ["pytorch_model.bin"]

    sharded_pytorch = [
        name
        for name in filenames
        if re.search(r"-\d{5}-of-\d{5}\.bin$", name)
    ]
    if sharded_pytorch:
        return sorted(sharded_pytorch)

    if "model.npz" in filename_set:
        return ["model.npz"]

    for suffix in HF_WEIGHT_SUFFIXES:
        matches = [name for name in filenames if name.endswith(suffix)]
        if matches:
            return [sorted(matches)[0]]

    return []


def parse_huggingface_reference(model_url: str) -> HuggingFaceReference:
    """Parse a huggingface.co model page or file URL."""

    parsed = urlparse(model_url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise CheckpointLoadError(f"Not a Hugging Face model URL: {model_url}")

    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if not parts:
        raise CheckpointLoadError(f"Hugging Face URL is missing a repo id: {model_url}")

    if parts[0] == "models":
        parts = parts[1:]

    for marker in ("blob", "resolve", "raw"):
        if marker in parts:
            marker_index = parts.index(marker)
            repo_id = "/".join(parts[:marker_index])
            if marker_index + 2 >= len(parts):
                raise CheckpointLoadError(
                    f"Hugging Face file URL is incomplete: {model_url}"
                )
            revision = parts[marker_index + 1]
            filename = "/".join(parts[marker_index + 2 :])
            return HuggingFaceReference(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
            )

    if "tree" in parts:
        marker_index = parts.index("tree")
        repo_id = "/".join(parts[:marker_index])
        revision = parts[marker_index + 1] if marker_index + 1 < len(parts) else None
        return HuggingFaceReference(repo_id=repo_id, revision=revision)

    repo_id = parts[0] if len(parts) == 1 else "/".join(parts[:2])
    return HuggingFaceReference(repo_id=repo_id)


def is_huggingface_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in {
        "huggingface.co",
        "www.huggingface.co",
    }


def _huggingface_client() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise CheckpointLoadError(
            "Hugging Face online models require huggingface-hub. Install it "
            "with `python -m pip install -e .`, or `python -m pip install "
            "huggingface-hub`."
        ) from exc

    return HfApi(), hf_hub_download


def _torch_load(
    checkpoint_path: Path,
    *,
    map_location: str,
    unsafe_load: bool,
) -> Any:
    if not checkpoint_path.exists():
        raise CheckpointLoadError(f"Checkpoint not found: {checkpoint_path}")

    try:
        import torch
    except ImportError as exc:
        raise CheckpointLoadError(
            "PyTorch is required to read .pth/.pt files. Install the optional "
            "dependency with `python -m pip install -e .[pytorch]`, or install "
            "a platform-specific PyTorch build from https://pytorch.org/."
        ) from exc

    try:
        if unsafe_load:
            return torch.load(checkpoint_path, map_location=map_location)
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        if unsafe_load:
            return torch.load(checkpoint_path, map_location=map_location)
        return torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        if not unsafe_load:
            raise CheckpointLoadError(
                "Failed to load checkpoint in safe weights-only mode. If this "
                "file is trusted and contains a pickled Python object, rerun "
                "with --unsafe-load."
            ) from exc
        raise CheckpointLoadError(f"Failed to load checkpoint: {exc}") from exc


def find_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Find the most likely state_dict inside a loaded checkpoint object."""

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    if isinstance(checkpoint, Mapping):
        for key in COMMON_STATE_DICT_KEYS:
            if key in checkpoint:
                value = checkpoint[key]
                if _looks_like_state_dict(value):
                    return value
                if isinstance(value, Mapping):
                    nested = _find_state_dict_depth_first(value, max_depth=3)
                    if nested is not None:
                        return nested

        nested = _find_state_dict_depth_first(checkpoint, max_depth=4)
        if nested is not None:
            return nested

    raise CheckpointLoadError(
        "Could not find a tensor state_dict inside this checkpoint."
    )


def summarize_state_dict(state_dict: Mapping[str, Any]) -> StateDictInfo:
    """Convert tensor-like state_dict values to TensorInfo objects."""

    result: StateDictInfo = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        info = tensor_info(value)
        if info is not None:
            result[key] = info

    if not result:
        raise CheckpointLoadError("The detected state_dict contains no tensors.")

    return dict(sorted(result.items()))


def tensor_info(value: Any) -> TensorInfo | None:
    """Return TensorInfo for torch tensors, parameters, buffers, and arrays."""

    shape = getattr(value, "shape", None)
    if shape is None and hasattr(value, "size"):
        try:
            shape = value.size()
        except TypeError:
            shape = None

    if shape is None:
        return None

    try:
        parsed_shape = tuple(int(dim) for dim in shape)
    except TypeError:
        return None

    dtype = getattr(value, "dtype", None)
    return TensorInfo(shape=parsed_shape, dtype=str(dtype) if dtype else None)


def strip_common_prefix(
    state_dict: StateDictInfo,
    prefix: str,
) -> StateDictInfo:
    """Strip a prefix only when every key has that prefix."""

    if not prefix:
        return state_dict

    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _find_state_dict_depth_first(
    value: Mapping[str, Any],
    *,
    max_depth: int,
) -> Mapping[str, Any] | None:
    if max_depth <= 0:
        return None

    for nested_value in value.values():
        if _looks_like_state_dict(nested_value):
            return nested_value
        if isinstance(nested_value, Mapping):
            found = _find_state_dict_depth_first(
                nested_value,
                max_depth=max_depth - 1,
            )
            if found is not None:
                return found
    return None


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False

    tensor_count = 0
    checked = 0
    for item_value in value.values():
        checked += 1
        if tensor_info(item_value) is not None:
            tensor_count += 1
        if checked >= 20:
            break

    return tensor_count > 0 and tensor_count >= max(1, checked // 2)
