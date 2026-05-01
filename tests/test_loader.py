from pathlib import Path

import numpy as np

from model_printer.loader import (
    CheckpointLoadError,
    HuggingFaceReference,
    TensorInfo,
    load_npz_state_dict,
    load_safetensors_state_dict,
    load_state_dict,
    normalize_npz_key,
    parse_huggingface_reference,
    select_huggingface_weight_files,
    summarize_huggingface_safetensors_metadata,
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


def test_load_safetensors_state_dict_reads_header_only():
    checkpoint = Path.cwd() / ".test_loader_model.safetensors"
    header = {
        "__metadata__": {"format": "pt"},
        "stem.conv.weight": {
            "dtype": "F32",
            "shape": [8, 3, 3, 3],
            "data_offsets": [0, 864],
        },
        "head.bias": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [864, 868],
        },
    }
    raw_header = bytes(str(header).replace("'", '"'), "utf-8")
    try:
        with checkpoint.open("wb") as file:
            file.write(len(raw_header).to_bytes(8, "little"))
            file.write(raw_header)

        state_dict = load_safetensors_state_dict(checkpoint)
    finally:
        checkpoint.unlink(missing_ok=True)

    assert state_dict["stem.conv.weight"] == TensorInfo((8, 3, 3, 3), "F32")
    assert state_dict["head.bias"] == TensorInfo((2,), "F16")


def test_parse_huggingface_repo_url():
    reference = parse_huggingface_reference(
        "https://huggingface.co/google-bert/bert-base-uncased"
    )

    assert reference == HuggingFaceReference(repo_id="google-bert/bert-base-uncased")


def test_parse_huggingface_file_url():
    reference = parse_huggingface_reference(
        "https://huggingface.co/org/model/blob/main/sub/model.safetensors"
    )

    assert reference.repo_id == "org/model"
    assert reference.revision == "main"
    assert reference.filename == "sub/model.safetensors"


def test_select_huggingface_weight_files_prefers_safetensors_shards():
    selected = select_huggingface_weight_files(
        [
            "config.json",
            "pytorch_model.bin",
            "model-00002-of-00002.safetensors",
            "model-00001-of-00002.safetensors",
        ]
    )

    assert selected == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_summarize_huggingface_safetensors_metadata():
    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    class FakeFile:
        tensors = {
            "layer.weight": FakeTensor([4, 5], "F32"),
            "layer.bias": {"shape": [4], "dtype": "F16"},
        }

    state_dict = summarize_huggingface_safetensors_metadata([FakeFile()])

    assert state_dict["layer.weight"] == TensorInfo((4, 5), "F32")
    assert state_dict["layer.bias"] == TensorInfo((4,), "F16")
