from xml.etree import ElementTree as ET

from model_printer.drawio import export_drawio
from model_printer.loader import TensorInfo
from model_printer.tree import build_model_tree, compact_tree


def test_export_drawio_contains_graph_model_child():
    state_dict = {
        "stem.conv.weight": TensorInfo((8, 3, 3, 3)),
        "stem.bn.weight": TensorInfo((8,)),
        "stem.bn.running_mean": TensorInfo((8,)),
        "head.weight": TensorInfo((2, 8)),
    }

    rendered = compact_tree(build_model_tree(state_dict).root)
    xml = export_drawio(rendered)
    root = ET.fromstring(xml)
    diagram = root.find("diagram")

    assert diagram is not None
    assert diagram.find("mxGraphModel") is not None
    assert "&lt;mxGraphModel" not in xml
    assert "Conv2d out=8, in=3, k=3x3" in xml
