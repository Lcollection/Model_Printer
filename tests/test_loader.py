from pathlib import Path

import numpy as np

from model_printer.loader import (
    CheckpointLoadError,
    TensorInfo,
    load_npz_state_dict,
    load_state_dict,
    normalize_npz_key,
)


def test_load_npz_state_dict_reads_array_metadata():
    checkpoint = Path.cwd() / ".test_loader_metadata.npz"
    try:
        np.savez(
            checkpoint,
            **{
                "stem/conv/weight": np.zeros((8, 3, 3, 3), dtype=np.float32),
                "head.weight": np.zeros((2, 8), dtype=np.float16),
            },
        )

        state_dict = load_npz_state_dict(checkpoint)
    finally:
        checkpoint.unlink(missing_ok=True)

    assert state_dict["stem.conv.weight"] == TensorInfo(
        shape=(8, 3, 3, 3),
        dtype="float32",
    )
    assert state_dict["head.weight"] == TensorInfo(
        shape=(2, 8),
        dtype="float16",
    )


def test_load_state_dict_routes_npz_files():
    checkpoint = Path.cwd() / ".test_loader_route.npz"
    try:
        np.savez(checkpoint, **{"module.head.weight": np.zeros((2, 8))})

        state_dict = load_state_dict(checkpoint)
    finally:
        checkpoint.unlink(missing_ok=True)

    assert list(state_dict) == ["head.weight"]
    assert state_dict["head.weight"].shape == (2, 8)


def test_normalize_npz_key_converts_slashes_to_dots():
    assert normalize_npz_key("/encoder/0/attention/weight") == (
        "encoder.0.attention.weight"
    )


def test_missing_pth_reports_missing_file_before_torch_import():
    missing = Path.cwd() / ".test_missing_model.pth"

    try:
        load_state_dict(missing)
    except CheckpointLoadError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("Expected CheckpointLoadError")
