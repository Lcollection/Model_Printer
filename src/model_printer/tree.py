"""Build and compact a model tree from state_dict tensor metadata."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from .loader import StateDictInfo, TensorInfo


PARAMETER_NAMES = {
    "weight",
    "bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "in_proj_weight",
    "in_proj_bias",
    "q_proj_weight",
    "k_proj_weight",
    "v_proj_weight",
    "out_proj_weight",
}


@dataclass
class ParameterInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str | None = None

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass
class LayerNode:
    name: str
    path: tuple[str, ...] = field(default_factory=tuple)
    children: "OrderedDict[str, LayerNode]" = field(default_factory=OrderedDict)
    parameters: list[ParameterInfo] = field(default_factory=list)

    @property
    def param_count(self) -> int:
        return sum(parameter.numel for parameter in self.parameters) + sum(
            child.param_count for child in self.children.values()
        )

    @property
    def direct_param_count(self) -> int:
        return sum(parameter.numel for parameter in self.parameters)


@dataclass
class RenderNode:
    title: str
    subtitle: str
    path: tuple[str, ...]
    param_count: int
    direct_param_count: int
    parameters: tuple[ParameterInfo, ...] = ()
    children: tuple["RenderNode", ...] = ()
    repeat_count: int = 1
    repeated_names: tuple[str, ...] = ()

    @property
    def is_repeated(self) -> bool:
        return self.repeat_count > 1


@dataclass
class ModelTree:
    root: LayerNode
    tensor_count: int

    @property
    def param_count(self) -> int:
        return self.root.param_count


def build_model_tree(state_dict: StateDictInfo) -> ModelTree:
    """Build a hierarchical module tree from normalized state_dict keys."""

    root = LayerNode(name="Model", path=())

    for key, tensor in state_dict.items():
        parts = tuple(part for part in key.split(".") if part)
        if not parts:
            continue

        node_parts, parameter_name = split_parameter_key(parts)
        node = root
        for part in node_parts:
            if part not in node.children:
                node.children[part] = LayerNode(
                    name=part,
                    path=node.path + (part,),
                )
            node = node.children[part]

        node.parameters.append(
            ParameterInfo(
                name=parameter_name,
                shape=tensor.shape,
                dtype=tensor.dtype,
            )
        )

    return ModelTree(root=root, tensor_count=len(state_dict))


def split_parameter_key(parts: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Split a state_dict key into module path and parameter name."""

    if len(parts) == 1:
        return (), parts[0]

    if parts[-1] in PARAMETER_NAMES:
        return parts[:-1], parts[-1]

    if len(parts) >= 2 and parts[-2] in {"_packed_params", "packed_params"}:
        return parts[:-1], parts[-1]

    return parts[:-1], parts[-1]


def compact_tree(root: LayerNode, *, min_repeat: int = 2) -> RenderNode:
    """Convert a LayerNode tree into a render tree with repeated siblings folded."""

    return _render_node(root, min_repeat=max(1, min_repeat))


def render_text(root: RenderNode, *, max_depth: int | None = None) -> str:
    """Render a compact tree as indented plain text."""

    lines: list[str] = []

    def walk(node: RenderNode, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return

        indent = "  " * depth
        repeat = f" x{node.repeat_count}" if node.repeat_count > 1 else ""
        params = format_param_count(node.param_count)
        suffix = f" [params={params}]"
        if node.repeat_count > 1:
            suffix = f" [params={params} each]"
        subtitle = f": {node.subtitle}" if node.subtitle else ""
        lines.append(f"{indent}{node.title}{repeat}{subtitle}{suffix}")

        if node.parameters and max_depth is None:
            for parameter in node.parameters:
                shape = format_shape(parameter.shape)
                lines.append(f"{indent}  - {parameter.name}: {shape}")

        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


def node_signature(node: RenderNode) -> tuple:
    """A stable signature used to detect repeated sibling modules."""

    parameter_signature = tuple(
        sorted((parameter.name, parameter.shape) for parameter in node.parameters)
    )
    child_signature = tuple(node_signature(child) for child in node.children)
    return (node.subtitle, parameter_signature, child_signature)


def infer_layer_summary(node: LayerNode) -> str:
    """Infer a short layer label from direct parameters and node name."""

    params = {parameter.name: parameter for parameter in node.parameters}
    weight = params.get("weight")
    bias = params.get("bias")
    lower_path = ".".join(node.path).lower()
    lower_name = node.name.lower()

    if _has_any(params, "running_mean", "running_var"):
        channels = _first_dim(params.get("running_mean")) or _first_dim(weight)
        if channels is not None:
            return f"BatchNorm channels={channels}"
        return "BatchNorm"

    if "in_proj_weight" in params:
        shape = params["in_proj_weight"].shape
        if len(shape) == 2:
            embed_dim = shape[1]
            heads_hint = _name_hint(lower_path, ("head", "heads"))
            if heads_hint:
                return f"MultiHeadAttention dim={embed_dim}, heads~{heads_hint}"
            return f"MultiHeadAttention dim={embed_dim}"
        return "MultiHeadAttention"

    if weight is None:
        if node.parameters:
            return "Parameters"
        return ""

    shape = weight.shape
    has_bias = bias is not None
    bias_text = ", bias" if has_bias else ""

    if len(shape) == 4:
        out_channels, in_channels, kernel_h, kernel_w = shape
        return (
            "Conv2d "
            f"out={out_channels}, in={in_channels}, "
            f"k={kernel_h}x{kernel_w}{bias_text}"
        )

    if len(shape) == 5:
        out_channels, in_channels, depth, kernel_h, kernel_w = shape
        return (
            "Conv3d "
            f"out={out_channels}, in={in_channels}, "
            f"k={depth}x{kernel_h}x{kernel_w}{bias_text}"
        )

    if len(shape) == 3 and (
        "conv1d" in lower_path or "temporal" in lower_path or "tdnn" in lower_path
    ):
        out_channels, in_channels, kernel = shape
        return f"Conv1d out={out_channels}, in={in_channels}, k={kernel}{bias_text}"

    if len(shape) == 2:
        rows, cols = shape
        if any(token in lower_path for token in ("embed", "embedding", "token")):
            return f"Embedding vocab={rows}, dim={cols}"
        if any(token in lower_name for token in ("qkv", "query_key_value")):
            return f"QKV Linear out={rows}, in={cols}{bias_text}"
        return f"Linear out={rows}, in={cols}{bias_text}"

    if len(shape) == 1:
        dim = shape[0]
        if any(token in lower_path for token in ("norm", "ln", "layernorm")):
            return f"LayerNorm dim={dim}"
        return f"Vector dim={dim}"

    if shape == ():
        return "Scalar"

    return f"Tensor {format_shape(shape)}"


def format_shape(shape: Iterable[int]) -> str:
    shape_tuple = tuple(shape)
    if not shape_tuple:
        return "()"
    return "x".join(str(dim) for dim in shape_tuple)


def format_param_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _render_node(node: LayerNode, *, min_repeat: int) -> RenderNode:
    rendered_children = [
        _render_node(child, min_repeat=min_repeat)
        for child in node.children.values()
    ]
    compact_children = tuple(_compact_siblings(rendered_children, min_repeat))

    return RenderNode(
        title=node.name,
        subtitle=infer_layer_summary(node),
        path=node.path,
        param_count=node.param_count,
        direct_param_count=node.direct_param_count,
        parameters=tuple(sorted(node.parameters, key=lambda item: item.name)),
        children=compact_children,
    )


def _compact_siblings(
    siblings: list[RenderNode],
    min_repeat: int,
) -> list[RenderNode]:
    if len(siblings) < min_repeat:
        return siblings

    compacted: list[RenderNode] = []
    index = 0
    while index < len(siblings):
        current = siblings[index]
        current_signature = node_signature(current)
        run = [current]
        index += 1

        while index < len(siblings) and node_signature(siblings[index]) == current_signature:
            run.append(siblings[index])
            index += 1

        if len(run) >= min_repeat:
            compacted.append(_merge_repeated_run(run))
        else:
            compacted.extend(run)

    return compacted


def _merge_repeated_run(run: list[RenderNode]) -> RenderNode:
    first = run[0]
    names = tuple(node.title for node in run)
    merged_title = _range_title(names)
    return RenderNode(
        title=merged_title,
        subtitle=first.subtitle,
        path=first.path,
        param_count=first.param_count,
        direct_param_count=first.direct_param_count,
        parameters=first.parameters,
        children=first.children,
        repeat_count=len(run),
        repeated_names=names,
    )


def _range_title(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    if all(name.isdigit() for name in names):
        return f"{names[0]}..{names[-1]}"
    return f"{names[0]}..{names[-1]}"


def _has_any(params: dict[str, ParameterInfo], *names: str) -> bool:
    return any(name in params for name in names)


def _first_dim(parameter: ParameterInfo | None) -> int | None:
    if parameter is None or not parameter.shape:
        return None
    return parameter.shape[0]


def _name_hint(text: str, tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        if token in text:
            return token
    return None
