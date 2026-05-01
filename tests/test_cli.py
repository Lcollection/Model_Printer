from model_printer.cli import build_parser
from model_printer.cli import default_output_path
from model_printer.cli import main


def test_checkpoint_argument_is_optional_for_welcome_screen():
    args = build_parser().parse_args([])

    assert args.checkpoint is None


def test_checkpoint_argument_still_accepts_file_path():
    args = build_parser().parse_args(["model.npz", "--tui"])

    assert args.checkpoint == "model.npz"
    assert args.tui is True


def test_no_checkpoint_in_non_interactive_session_prints_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    exit_code = main([])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Vim-like welcome" in output
    assert "screen" in output


def test_default_output_path_for_huggingface_repo_url():
    output = default_output_path("https://huggingface.co/google-bert/bert-base-uncased")

    assert output.name == "bert-base-uncased.drawio"
