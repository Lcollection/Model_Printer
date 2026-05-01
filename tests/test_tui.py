from model_printer.loader import TensorInfo
from model_printer.tree import build_model_tree, compact_tree
from model_printer.tui import (
    WelcomeState,
    WELCOME_ASCII,
    WELCOME_TITLE,
    collect_all_expandable_paths,
    collect_default_expanded_paths,
    execute_welcome_command,
    flatten_visible_nodes,
    is_url_text,
    scroll_start,
)


def _sample_render_tree():
    state_dict = {
        "stem.conv.weight": TensorInfo((8, 3, 3, 3)),
        "stage.0.conv.weight": TensorInfo((8, 8, 3, 3)),
        "stage.1.conv.weight": TensorInfo((8, 8, 3, 3)),
        "head.weight": TensorInfo((2, 8)),
    }
    return compact_tree(build_model_tree(state_dict).root, min_repeat=2)


def test_default_expansion_keeps_initial_tree_readable():
    root = _sample_render_tree()
    expanded = collect_default_expanded_paths(root, max_depth=1)
    visible = flatten_visible_nodes(root, expanded)
    titles = [item.node.title for item in visible]

    assert "Model" in titles
    assert "stem" in titles
    assert "stage" in titles
    assert "head" in titles
    assert "conv" not in titles


def test_expand_all_includes_nested_children():
    root = _sample_render_tree()
    visible = flatten_visible_nodes(root, collect_all_expandable_paths(root))
    titles = [item.node.title for item in visible]

    assert "conv" in titles
    assert "0..1" in titles


def test_scroll_start_tracks_selection_without_overflow():
    assert scroll_start(selected_index=0, total=100, available_height=10) == 0
    assert scroll_start(selected_index=50, total=100, available_height=10) == 45
    assert scroll_start(selected_index=99, total=100, available_height=10) == 90


def test_welcome_open_command_returns_checkpoint_path():
    state = WelcomeState(command=":open demo/model.npz", command_mode=True)

    result = execute_welcome_command(state)

    assert result is not None
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.as_posix().endswith("demo/model.npz")


def test_welcome_open_command_keeps_url_as_text():
    state = WelcomeState(
        command=":open https://huggingface.co/google-bert/bert-base-uncased",
        command_mode=True,
    )

    result = execute_welcome_command(state)

    assert result is not None
    assert result.checkpoint_path == "https://huggingface.co/google-bert/bert-base-uncased"


def test_welcome_help_command_stays_on_splash():
    state = WelcomeState(command=":help", command_mode=True)

    result = execute_welcome_command(state)

    assert result is None
    assert state.command_mode is False
    assert "Commands:" in state.message


def test_welcome_ascii_banner_spells_model_printer():
    banner = "\n".join(WELCOME_ASCII)

    assert WELCOME_TITLE == "Model printer"
    assert len(WELCOME_ASCII) == 6
    assert "__  __" in banner
    assert "_ __  _ __" in banner


def test_is_url_text_detects_http_urls():
    assert is_url_text("https://huggingface.co/org/model")
    assert not is_url_text("E:/models/model.npz")
