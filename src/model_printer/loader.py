"""Checkpoint loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def load_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    unsafe_load: bool = False,
    strip_module_prefix: bool = True,
    strip_prefixes: tuple[str, ...] = (),
) -> StateDictInfo:
    """Load a PyTorch checkpoint and return a normalized state_dict summary.

    The returned dictionary contains only tensor metadata. Keeping metadata
    instead of tensors avoids carrying a large model in memory through the rest
    of the rendering pipeline.
    """

    path = Path(checkpoint_path)
    if path.suffix.lower() == ".npz":
        normalized = load_npz_state_dict(path)
    else:
        checkpoint = _torch_load(
            path,
            map_location=map_location,
            unsafe_load=unsafe_load,
        )
        state_dict = find_state_dict(checkpoint)
        normalized = summarize_state_dict(state_dict)

    if strip_module_prefix:
        normalized = strip_common_prefix(normalized, "module.")

    for prefix in strip_prefixes:
        normalized = strip_common_prefix(normalized, prefix)

    return normalized


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


def normalize_npz_key(key: str) -> str:
    """Normalize common NPZ path separators to state_dict-style dotted keys."""

    return key.replace("/", ".").strip(".")


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
