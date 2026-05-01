"""Interactive terminal UI for browsing compact model trees."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .drawio import export_drawio
from .tree import RenderNode, format_param_count, format_shape


WELCOME_TITLE = "Model printer"
WELCOME_ASCII = (
    r" __  __           _      _             _       _            ",
    r"|  \/  | ___   __| | ___| |  _ __  _ __(_)_ __ | |_ ___ _ __ ",
    r"| |\/| |/ _ \ / _` |/ _ \ | | '_ \| '__| | '_ \| __/ _ \ '__|",
    r"| |  | | (_) | (_| |  __/ | | |_) | |  | | | | | ||  __/ |   ",
    r"|_|  |_|\___/ \__,_|\___|_| | .__/|_|  |_|_| |_|\__\___|_|   ",
    r"                            |_|                              ",
)


@dataclass(frozen=True)
class VisibleNode:
    node: RenderNode
    depth: int


@dataclass
class TuiState:
    selected_index: int
    expanded_paths: set[tuple[str, ...]]
    message: str
    exported_path: Path | None = None


@dataclass(frozen=True)
class WelcomeResult:
    checkpoint_path: str | Path | None = None


@dataclass
class WelcomeState:
    command: str = ""
    command_mode: bool = False
    message: str = "Press : to enter a command, or q to quit"


def run_welcome_screen() -> WelcomeResult:
    """Run a Vim-like opening screen when no checkpoint is provided."""

    console = Console()
    state = WelcomeState()

    def render() -> RenderableType:
        return render_welcome_screen(
            state,
            console_width=console.size.width,
            console_height=console.size.height,
        )

    with Live(render(), console=console, screen=True, refresh_per_second=12) as live:
        while True:
            key = read_key()

            if state.command_mode:
                result = handle_welcome_command_key(state, key)
                if result is not None:
                    return result
            elif key in {"q", "esc", "ctrl+c"}:
                return WelcomeResult()
            elif key == ":":
                state.command_mode = True
                state.command = ":"
                state.message = "Command mode"
            elif key == "o":
                state.command_mode = True
                state.command = ":open "
                state.message = "Type a checkpoint path, then press Enter"
            elif key in {"h", "?"}:
                state.message = "Use :open <path>, :q, or q"
            else:
                state.message = f"Unknown key: {key}"

            live.update(render())


def run_tui(
    root: RenderNode,
    *,
    checkpoint_path: Path,
    output_path: Path,
    tensor_count: int,
) -> int:
    """Run the interactive terminal browser."""

    console = Console()
    state = TuiState(
        selected_index=0,
        expanded_paths=collect_default_expanded_paths(root, max_depth=2),
        message="Ready",
    )

    def render() -> RenderableType:
        visible_nodes = flatten_visible_nodes(root, state.expanded_paths)
        state.selected_index = clamp_index(state.selected_index, visible_nodes)
        return render_screen(
            root,
            visible_nodes,
            state,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            tensor_count=tensor_count,
            console_height=console.size.height,
        )

    with Live(render(), console=console, screen=True, refresh_per_second=12) as live:
        while True:
            key = read_key()
            visible_nodes = flatten_visible_nodes(root, state.expanded_paths)
            state.selected_index = clamp_index(state.selected_index, visible_nodes)

            if key in {"q", "esc", "ctrl+c"}:
                break
            if key in {"up", "k"}:
                state.selected_index = max(0, state.selected_index - 1)
            elif key in {"down", "j"}:
                state.selected_index = min(
                    len(visible_nodes) - 1,
                    state.selected_index + 1,
                )
            elif key == "home":
                state.selected_index = 0
            elif key == "end":
                state.selected_index = len(visible_nodes) - 1
            elif key in {"pageup", "u"}:
                state.selected_index = max(0, state.selected_index - 10)
            elif key in {"pagedown", "d"}:
                state.selected_index = min(
                    len(visible_nodes) - 1,
                    state.selected_index + 10,
                )
            elif key in {"enter", "space", "right", "l"}:
                toggle_selected(visible_nodes, state)
            elif key in {"left", "h"}:
                collapse_or_select_parent(visible_nodes, state)
            elif key == "a":
                state.expanded_paths = collect_all_expandable_paths(root)
                state.message = "Expanded all nodes"
            elif key == "c":
                state.expanded_paths = {root.path}
                state.selected_index = 0
                state.message = "Collapsed tree"
            elif key == "e":
                state.message = export_from_tui(root, output_path)
                state.exported_path = output_path
            else:
                state.message = f"Unknown key: {key}"

            live.update(render())

    if state.exported_path is not None:
        console.print(f"drawio: {state.exported_path}")
    return 0


def flatten_visible_nodes(
    root: RenderNode,
    expanded_paths: set[tuple[str, ...]],
) -> list[VisibleNode]:
    """Return nodes visible in the tree pane according to expansion state."""

    rows: list[VisibleNode] = []

    def walk(node: RenderNode, depth: int) -> None:
        rows.append(VisibleNode(node=node, depth=depth))
        if node.path not in expanded_paths:
            return
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    return rows


def collect_default_expanded_paths(
    root: RenderNode,
    *,
    max_depth: int,
) -> set[tuple[str, ...]]:
    """Expand the first few levels by default."""

    expanded: set[tuple[str, ...]] = set()

    def walk(node: RenderNode, depth: int) -> None:
        if node.children and depth < max_depth:
            expanded.add(node.path)
            for child in node.children:
                walk(child, depth + 1)

    walk(root, 0)
    return expanded


def collect_all_expandable_paths(root: RenderNode) -> set[tuple[str, ...]]:
    expanded: set[tuple[str, ...]] = set()

    def walk(node: RenderNode) -> None:
        if node.children:
            expanded.add(node.path)
            for child in node.children:
                walk(child)

    walk(root)
    return expanded


def render_screen(
    root: RenderNode,
    visible_nodes: list[VisibleNode],
    state: TuiState,
    *,
    checkpoint_path: Path,
    output_path: Path,
    tensor_count: int,
    console_height: int,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="tree", ratio=3, minimum_size=42),
        Layout(name="detail", ratio=2, minimum_size=36),
    )

    selected = visible_nodes[state.selected_index].node
    layout["header"].update(
        render_header(
            root,
            checkpoint_path=checkpoint_path,
            tensor_count=tensor_count,
        )
    )
    layout["tree"].update(
        render_tree_pane(
            visible_nodes,
            selected_index=state.selected_index,
            expanded_paths=state.expanded_paths,
            available_height=max(8, console_height - 8),
        )
    )
    layout["detail"].update(render_detail_pane(selected, output_path=output_path))
    layout["footer"].update(render_footer(state.message))
    return layout


def render_welcome_screen(
    state: WelcomeState,
    *,
    console_width: int,
    console_height: int,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="body"),
        Layout(name="command", size=3),
    )
    layout["body"].update(render_welcome_body(console_width, console_height))
    layout["command"].update(render_welcome_command(state))
    return layout


def render_welcome_body(console_width: int, console_height: int) -> Panel:
    visible_lines = max(12, console_height - 6)
    center_row = max(2, visible_lines // 2 - 8)
    text = Text()

    welcome_lines: dict[int, tuple[str, str]] = {}
    for offset, line in enumerate(WELCOME_ASCII):
        welcome_lines[center_row + offset] = (line, "bold cyan")

    prompt_row = center_row + len(WELCOME_ASCII) + 2
    welcome_lines.update(
        {
            prompt_row: (WELCOME_TITLE, "white"),
            prompt_row + 1: ("Model structure viewer", "white"),
            prompt_row + 2: (":open <checkpoint>   open .pth/.pt/.npz", "green"),
            prompt_row + 3: ("o                    shortcut for :open", "green"),
            prompt_row + 4: (":help                show command help", "yellow"),
            prompt_row + 5: (":q or q              quit", "yellow"),
            prompt_row + 7: ("https://github.com/Lcollection/Model_Printer", "blue"),
        }
    )

    for row in range(visible_lines):
        if row in welcome_lines:
            label, style = welcome_lines[row]
            text.append(center_text(label, console_width - 4), style=style)
        else:
            text.append("~", style="dim cyan")
        if row < visible_lines - 1:
            text.append("\n")

    return Panel(text, border_style="cyan")


def render_welcome_command(state: WelcomeState) -> Panel:
    prompt = Text()
    if state.command_mode:
        prompt.append(state.command or ":", style="bold white")
    else:
        prompt.append(state.message, style="cyan")
    return Panel(prompt, border_style="cyan")


def render_header(
    root: RenderNode,
    *,
    checkpoint_path: Path,
    tensor_count: int,
) -> Panel:
    title = Text("Model Printer TUI", style="bold cyan")
    summary = Text()
    summary.append("  ")
    summary.append(str(checkpoint_path), style="white")
    summary.append("  |  ")
    summary.append(f"{tensor_count} tensors", style="green")
    summary.append("  |  ")
    summary.append(f"{format_param_count(root.param_count)} parameters", style="yellow")
    return Panel(Group(title, summary), border_style="cyan")


def render_tree_pane(
    visible_nodes: list[VisibleNode],
    *,
    selected_index: int,
    expanded_paths: set[tuple[str, ...]],
    available_height: int,
) -> Panel:
    text = Text()
    if not visible_nodes:
        return Panel(Align.center("No nodes"), title="Structure", border_style="blue")

    start = scroll_start(selected_index, len(visible_nodes), available_height)
    end = min(len(visible_nodes), start + available_height)

    for index in range(start, end):
        visible = visible_nodes[index]
        line = format_tree_line(
            visible,
            selected=index == selected_index,
            expanded_paths=expanded_paths,
        )
        text.append_text(line)
        if index < end - 1:
            text.append("\n")

    title = f"Structure {selected_index + 1}/{len(visible_nodes)}"
    return Panel(text, title=title, border_style="blue")


def render_detail_pane(node: RenderNode, *, output_path: Path) -> Panel:
    details = Table.grid(expand=True)
    details.add_column("name", style="bold")
    details.add_column("value")
    details.add_row("Name", _title_with_repeat(node))
    details.add_row("Path", ".".join(node.path) if node.path else "<root>")
    details.add_row("Type", node.subtitle or "Module")
    details.add_row("Params", format_param_count(node.param_count))
    details.add_row("Direct", format_param_count(node.direct_param_count))
    details.add_row("Children", str(len(node.children)))
    if node.repeat_count > 1:
        details.add_row("Repeat", f"{node.repeat_count} similar layers")
        details.add_row("Members", _format_repeated_names(node.repeated_names))

    parameter_table = Table(
        title="Parameters",
        expand=True,
        show_header=True,
        header_style="bold magenta",
    )
    parameter_table.add_column("name", no_wrap=True)
    parameter_table.add_column("shape", no_wrap=True)
    parameter_table.add_column("count", justify="right", no_wrap=True)
    parameter_table.add_column("dtype", overflow="fold")

    if node.parameters:
        for parameter in node.parameters:
            parameter_table.add_row(
                parameter.name,
                format_shape(parameter.shape),
                format_param_count(parameter.numel),
                parameter.dtype or "",
            )
    else:
        parameter_table.add_row("-", "-", "-", "")

    child_table = Table(
        title="Children",
        expand=True,
        show_header=True,
        header_style="bold green",
    )
    child_table.add_column("name", no_wrap=True)
    child_table.add_column("type")
    child_table.add_column("params", justify="right", no_wrap=True)
    for child in node.children[:8]:
        child_table.add_row(
            _title_with_repeat(child),
            child.subtitle or "Module",
            format_param_count(child.param_count),
        )
    if len(node.children) > 8:
        child_table.add_row(f"+{len(node.children) - 8} more", "", "")
    if not node.children:
        child_table.add_row("-", "-", "-")

    output = Text()
    output.append("Export target: ", style="bold")
    output.append(str(output_path), style="cyan")

    return Panel(
        Group(details, Text(), parameter_table, Text(), child_table, Text(), output),
        title="Selected Layer",
        border_style="magenta",
    )


def render_footer(message: str) -> Panel:
    keys = Text()
    keys.append("Up/Down or j/k", style="bold")
    keys.append(" move  ")
    keys.append("Enter", style="bold")
    keys.append(" toggle  ")
    keys.append("Left/Right", style="bold")
    keys.append(" fold/open  ")
    keys.append("a", style="bold")
    keys.append(" all  ")
    keys.append("c", style="bold")
    keys.append(" collapse  ")
    keys.append("e", style="bold")
    keys.append(" export draw.io  ")
    keys.append("q", style="bold")
    keys.append(" quit")

    status = Text(message, style="cyan")
    return Panel(Group(keys, status), border_style="cyan")


def format_tree_line(
    visible: VisibleNode,
    *,
    selected: bool,
    expanded_paths: set[tuple[str, ...]],
) -> Text:
    node = visible.node
    marker = "> " if selected else "  "
    if node.children:
        toggle = "- " if node.path in expanded_paths else "+ "
    else:
        toggle = "  "

    repeat = f" x{node.repeat_count}" if node.repeat_count > 1 else ""
    params = format_param_count(node.param_count)
    subtitle = f"  {node.subtitle}" if node.subtitle else ""
    line = Text()
    style = "reverse bold" if selected else ""
    line.append(marker, style=style)
    line.append("  " * visible.depth, style=style)
    line.append(toggle, style=style)
    line.append(f"{node.title}{repeat}", style=f"bold {style}".strip())
    line.append(subtitle, style=style)
    line.append(f"  [{params}]", style=("yellow " + style).strip())
    return line


def toggle_selected(visible_nodes: list[VisibleNode], state: TuiState) -> None:
    node = visible_nodes[state.selected_index].node
    if not node.children:
        state.message = "Selected node has no children"
        return
    if node.path in state.expanded_paths:
        state.expanded_paths.remove(node.path)
        state.message = f"Collapsed {_title_with_repeat(node)}"
    else:
        state.expanded_paths.add(node.path)
        state.message = f"Expanded {_title_with_repeat(node)}"


def collapse_or_select_parent(
    visible_nodes: list[VisibleNode],
    state: TuiState,
) -> None:
    node = visible_nodes[state.selected_index].node
    if node.children and node.path in state.expanded_paths:
        state.expanded_paths.remove(node.path)
        state.message = f"Collapsed {_title_with_repeat(node)}"
        return

    if not node.path:
        state.message = "Already at root"
        return

    parent_path = node.path[:-1]
    for index, visible in enumerate(visible_nodes):
        if visible.node.path == parent_path:
            state.selected_index = index
            state.message = f"Selected parent {_title_with_repeat(visible.node)}"
            return


def export_from_tui(root: RenderNode, output_path: Path) -> str:
    try:
        output_path.write_text(export_drawio(root), encoding="utf-8")
    except OSError as exc:
        return f"Export failed: {exc}"
    return f"Exported draw.io to {output_path}"


def handle_welcome_command_key(
    state: WelcomeState,
    key: str,
) -> WelcomeResult | None:
    if key in {"esc", "ctrl+c"}:
        state.command_mode = False
        state.command = ""
        state.message = "Command canceled"
        return None
    if key == "backspace":
        if len(state.command) > 1:
            state.command = state.command[:-1]
        return None
    if key == "enter":
        return execute_welcome_command(state)
    if len(key) == 1:
        state.command += key
        return None

    state.message = f"Unsupported command key: {key}"
    return None


def execute_welcome_command(state: WelcomeState) -> WelcomeResult | None:
    raw_command = state.command.strip()
    command = raw_command[1:].strip() if raw_command.startswith(":") else raw_command

    if command in {"q", "quit", "q!"}:
        return WelcomeResult()
    if command in {"h", "help", "?"}:
        state.command_mode = False
        state.command = ""
        state.message = "Commands: :open <path>, :q, :help"
        return None

    for prefix in ("open ", "edit ", "e "):
        if command.startswith(prefix):
            path_text = command[len(prefix) :].strip().strip("\"'")
            if not path_text:
                state.message = "Missing checkpoint path"
                return None
            if is_url_text(path_text):
                return WelcomeResult(checkpoint_path=path_text)
            return WelcomeResult(checkpoint_path=Path(path_text).expanduser())

    if command:
        state.command_mode = False
        state.command = ""
        state.message = f"Not an editor command: {raw_command}"
        return None

    state.command_mode = False
    state.command = ""
    state.message = "Empty command"
    return None


def read_key() -> str:
    if os.name == "nt":
        return _read_windows_key()
    return _read_posix_key()


def _read_windows_key() -> str:
    import msvcrt

    char = msvcrt.getwch()
    if char in {"\x00", "\xe0"}:
        code = msvcrt.getwch()
        return {
            "H": "up",
            "P": "down",
            "K": "left",
            "M": "right",
            "G": "home",
            "O": "end",
            "I": "pageup",
            "Q": "pagedown",
        }.get(code, code)
    if char == "\r":
        return "enter"
    if char == " ":
        return "space"
    if char == "\x1b":
        return "esc"
    if char == "\x03":
        return "ctrl+c"
    if char in {"\b", "\x7f"}:
        return "backspace"
    return char.lower()


def _read_posix_key() -> str:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":
            sequence = ""
            while select.select([sys.stdin], [], [], 0.01)[0]:
                sequence += sys.stdin.read(1)
            return {
                "[A": "up",
                "[B": "down",
                "[D": "left",
                "[C": "right",
                "[H": "home",
                "[F": "end",
                "[5~": "pageup",
                "[6~": "pagedown",
            }.get(sequence, "esc")
        if char in {"\r", "\n"}:
            return "enter"
        if char == " ":
            return "space"
        if char == "\x03":
            return "ctrl+c"
        if char in {"\x7f", "\b"}:
            return "backspace"
        return char.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def scroll_start(selected_index: int, total: int, available_height: int) -> int:
    if total <= available_height:
        return 0
    half = max(1, available_height // 2)
    start = selected_index - half
    return max(0, min(start, total - available_height))


def clamp_index(index: int, visible_nodes: Iterable[VisibleNode]) -> int:
    size = len(list(visible_nodes))
    if size == 0:
        return 0
    return max(0, min(index, size - 1))


def _title_with_repeat(node: RenderNode) -> str:
    if node.repeat_count > 1:
        return f"{node.title} x{node.repeat_count}"
    return node.title


def _format_repeated_names(names: tuple[str, ...]) -> str:
    if not names:
        return "-"
    if len(names) <= 6:
        return ", ".join(names)
    return f"{', '.join(names[:3])}, ... , {', '.join(names[-2:])}"


def center_text(value: str, width: int) -> str:
    if width <= len(value):
        return value
    return value.center(width)


def is_url_text(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
