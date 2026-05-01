from model_printer.loader import TensorInfo
from model_printer.tree import build_model_tree, compact_tree, render_text


def test_build_tree_and_compact_repeated_blocks():
    state_dict = {
        "module.layer.0.conv.weight": TensorInfo((16, 16, 3, 3)),
        "module.layer.0.bn.weight": TensorInfo((16,)),
        "module.layer.0.bn.running_mean": TensorInfo((16,)),
        "module.layer.1.conv.weight": TensorInfo((16, 16, 3, 3)),
        "module.layer.1.bn.weight": TensorInfo((16,)),
        "module.layer.1.bn.running_mean": TensorInfo((16,)),
        "module.head.weight": TensorInfo((10, 16)),
        "module.head.bias": TensorInfo((10,)),
    }

    tree = build_model_tree(
        {key.removeprefix("module."): value for key, value in state_dict.items()}
    )
    rendered = compact_tree(tree.root, min_repeat=2)
    text = render_text(rendered)

    assert "0..1 x2" in text
    assert "Conv2d out=16, in=16, k=3x3" in text
    assert "Linear out=10, in=16, bias" in text


def test_no_compaction_for_different_shapes():
    state_dict = {
        "layer.0.conv.weight": TensorInfo((16, 8, 3, 3)),
        "layer.1.conv.weight": TensorInfo((32, 16, 3, 3)),
    }

    rendered = compact_tree(build_model_tree(state_dict).root, min_repeat=2)
    text = render_text(rendered)

    assert "0..1 x2" not in text
    assert "    0 [params=" in text
    assert "    1 [params=" in text
