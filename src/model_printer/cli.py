"""Command line interface for Model Printer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .drawio import export_drawio
from .loader import CheckpointLoadError, load_state_dict
from .tree import build_model_tree, compact_tree, format_param_count, render_text


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.checkpoint is None:
        if not sys.stdin.isatty():
            parser.print_help()
            return 0

        from .tui import run_welcome_screen

        welcome_result = run_welcome_screen()
        if welcome_result.checkpoint_path is None:
            return 0
        args.checkpoint = welcome_result.checkpoint_path
        args.tui = True

    try:
        state_dict = load_state_dict(
            args.checkpoint,
            unsafe_load=args.unsafe_load,
            strip_module_prefix=not args.keep_module_prefix,
            strip_prefixes=tuple(args.strip_prefix),
        )
    except CheckpointLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model_tree = build_model_tree(state_dict)
    render_root = compact_tree(model_tree.root, min_repeat=args.min_repeat)

    if args.tui:
        from .tui import run_tui

        output_path = args.output or default_output_path(args.checkpoint)
        return run_tui(
            render_root,
            checkpoint_path=args.checkpoint,
            output_path=output_path,
            tensor_count=model_tree.tensor_count,
        )

    print(render_text(render_root, max_depth=args.max_depth))
    print()
    print(
        "summary: "
        f"{model_tree.tensor_count} tensors, "
        f"{format_param_count(model_tree.param_count)} parameters"
    )

    if not args.no_drawio:
        output_path = args.output or default_output_path(args.checkpoint)
        output_path.write_text(export_drawio(render_root), encoding="utf-8")
        print(f"drawio: {output_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_printer",
        description=(
            "Inspect PyTorch .pth or NumPy .npz checkpoints and export compact draw.io "
            "architecture diagrams."
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help=(
            "Path to a .pth/.pt checkpoint, state_dict file, or .npz archive. "
            "Omit it to open the Vim-like welcome screen."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the generated .drawio file. Defaults to checkpoint name.",
    )
    parser.add_argument(
        "--no-drawio",
        action="store_true",
        help="Only print the model tree; do not write a .drawio file.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Open an interactive terminal UI for browsing layers and exporting "
            "draw.io."
        ),
    )
    parser.add_argument(
        "--min-repeat",
        type=int,
        default=2,
        help="Minimum number of consecutive identical sibling layers to fold.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit printed tree depth. The draw.io output is always complete.",
    )
    parser.add_argument(
        "--keep-module-prefix",
        action="store_true",
        help="Keep the common DataParallel 'module.' prefix if present.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help=(
            "Strip a prefix when all state_dict keys share it. Can be passed "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help=(
            "Allow PyTorch pickle loading for trusted checkpoints that cannot "
            "be read in weights-only mode."
        ),
    )
    return parser


def default_output_path(checkpoint_path: Path) -> Path:
    suffix = checkpoint_path.suffix
    if suffix:
        return checkpoint_path.with_suffix(".drawio")
    return checkpoint_path.with_name(f"{checkpoint_path.name}.drawio")


if __name__ == "__main__":
    raise SystemExit(main())
