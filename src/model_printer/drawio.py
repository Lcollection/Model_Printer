"""draw.io XML exporter."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from itertools import count
from xml.etree import ElementTree as ET

from .tree import RenderNode, format_param_count, format_shape


@dataclass
class LayoutBox:
    id: str
    node: RenderNode
    x: int
    y: int
    width: int
    height: int
    parent_id: str


def export_drawio(root: RenderNode) -> str:
    """Return a diagrams.net/draw.io XML document for a compact model tree."""

    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-05-02T00:00:00.000Z",
            "agent": "Model Printer",
            "version": "24.0.0",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"name": "Model Architecture"})
    graph = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1200",
            "dy": "900",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1400",
            "pageHeight": "1800",
            "math": "0",
            "shadow": "0",
        },
    )
    graph_root = ET.SubElement(graph, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})

    id_counter = count(2)
    boxes: list[LayoutBox] = []
    edges: list[tuple[str, str]] = []

    _layout(
        root,
        x=40,
        y=40,
        depth=0,
        parent_cell_id="1",
        id_counter=id_counter,
        boxes=boxes,
        edges=edges,
    )

    for box in boxes:
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": box.id,
                "value": _node_label(box.node),
                "style": _node_style(box.node),
                "vertex": "1",
                "parent": box.parent_id,
            },
        )
        geometry = ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(box.x),
                "y": str(box.y),
                "width": str(box.width),
                "height": str(box.height),
                "as": "geometry",
            },
        )
        geometry.text = None

    for source_id, target_id in edges:
        edge_id = str(next(id_counter))
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": edge_id,
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
                    "jettySize=auto;html=1;strokeColor=#586174;strokeWidth=1.5;"
                    "endArrow=block;endFill=1;"
                ),
                "edge": "1",
                "parent": "1",
                "source": source_id,
                "target": target_id,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    return ET.tostring(mxfile, encoding="unicode", xml_declaration=True)


def _layout(
    node: RenderNode,
    *,
    x: int,
    y: int,
    depth: int,
    parent_cell_id: str,
    id_counter: count,
    boxes: list[LayoutBox],
    edges: list[tuple[str, str]],
) -> tuple[int, int, str]:
    node_id = str(next(id_counter))
    width = 330
    height = _node_height(node)

    children_start_y = y + height + 36
    child_y = children_start_y
    child_ids: list[str] = []
    max_child_right = x + width

    for child in node.children:
        child_bottom, child_right, child_id = _layout(
            child,
            x=x + 390,
            y=child_y,
            depth=depth + 1,
            parent_cell_id="1",
            id_counter=id_counter,
            boxes=boxes,
            edges=edges,
        )
        child_ids.append(child_id)
        child_y = child_bottom + 24
        max_child_right = max(max_child_right, child_right)

    if child_ids:
        subtree_height = max(child_y - 24 - y, height)
        node_y = y + max(0, (subtree_height - height) // 2)
    else:
        subtree_height = height
        node_y = y

    boxes.append(
        LayoutBox(
            id=node_id,
            node=node,
            x=x,
            y=node_y,
            width=width,
            height=height,
            parent_id=parent_cell_id,
        )
    )

    for child_id in child_ids:
        edges.append((node_id, child_id))

    bottom = y + subtree_height
    return bottom, max_child_right, node_id


def _node_label(node: RenderNode) -> str:
    title = escape(node.title)
    if node.repeat_count > 1:
        title = f"{title} x{node.repeat_count}"

    lines = [f"<b>{title}</b>"]
    if node.subtitle:
        lines.append(escape(node.subtitle))

    params_text = format_param_count(node.param_count)
    if node.repeat_count > 1:
        lines.append(f"params: {params_text} each")
    else:
        lines.append(f"params: {params_text}")

    for parameter in node.parameters[:3]:
        lines.append(
            f"{escape(parameter.name)}: {escape(format_shape(parameter.shape))}"
        )
    if len(node.parameters) > 3:
        lines.append(f"+{len(node.parameters) - 3} params")

    return "<br>".join(lines)


def _node_style(node: RenderNode) -> str:
    fill = "#f8fafc"
    stroke = "#64748b"
    font = "#0f172a"

    if not node.path:
        fill = "#e0f2fe"
        stroke = "#0284c7"
    elif node.repeat_count > 1:
        fill = "#ecfdf5"
        stroke = "#059669"
    elif "Conv" in node.subtitle:
        fill = "#eff6ff"
        stroke = "#2563eb"
    elif "Linear" in node.subtitle or "Embedding" in node.subtitle:
        fill = "#fef3c7"
        stroke = "#d97706"
    elif "Norm" in node.subtitle:
        fill = "#f5f3ff"
        stroke = "#7c3aed"

    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={fill};strokeColor={stroke};fontColor={font};"
        "fontSize=12;align=left;verticalAlign=middle;spacingLeft=12;"
        "spacingRight=12;spacingTop=8;spacingBottom=8;"
    )


def _node_height(node: RenderNode) -> int:
    line_count = 2 + min(len(node.parameters), 3)
    if node.subtitle:
        line_count += 1
    if len(node.parameters) > 3:
        line_count += 1
    return max(86, 26 + line_count * 18)
