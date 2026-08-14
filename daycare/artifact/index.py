"""Read/write the <tensors> XML element (tensor-index.xml structure).

This module owns the <tensors> element only -- the <adapter>/<model> wrapper
around it lives in manifest.py (owned elsewhere).
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: tuple
    quant: str = None
    scale: float = None
    encoding: str = "external"
    # external variant
    file: str = None
    offset: int = None
    length: int = None
    # base64-inline variant
    data: str = None


def to_element(specs, encoding: str = "external") -> ET.Element:
    """[TensorSpec] -> <tensors> Element."""
    root = ET.Element("tensors", {"encoding": encoding})
    for s in specs:
        attrib = {
            "name": s.name,
            "dtype": s.dtype,
            "shape": ",".join(str(d) for d in s.shape),
        }
        if s.quant is not None:
            attrib["quant"] = s.quant
        if s.scale is not None:
            attrib["scale"] = repr(s.scale)
        if encoding == "external":
            attrib["file"] = s.file
            attrib["offset"] = str(s.offset)
            attrib["length"] = str(s.length)
            ET.SubElement(root, "tensor", attrib)
        else:  # base64
            t = ET.SubElement(root, "tensor", attrib)
            data_el = ET.SubElement(t, "data")
            data_el.text = s.data
    return root


def from_element(root: ET.Element):
    """<tensors> Element -> (encoding, [TensorSpec])."""
    encoding = root.get("encoding", "external")
    specs = []
    for t in root.findall("tensor"):
        shape = tuple(int(x) for x in t.get("shape").split(","))
        scale = t.get("scale")
        scale = float(scale) if scale is not None else None
        kwargs = dict(
            name=t.get("name"), dtype=t.get("dtype"), shape=shape,
            quant=t.get("quant"), scale=scale, encoding=encoding,
        )
        if encoding == "external":
            kwargs["file"] = t.get("file")
            kwargs["offset"] = int(t.get("offset"))
            kwargs["length"] = int(t.get("length"))
        else:
            data_el = t.find("data")
            kwargs["data"] = data_el.text if data_el is not None else None
        specs.append(TensorSpec(**kwargs))
    return encoding, specs


def write_xml(specs, encoding: str = "external", path: str = None) -> str:
    """[TensorSpec] -> xml string; optionally write to path."""
    xml_str = ET.tostring(to_element(specs, encoding), encoding="unicode")
    if path:
        with open(path, "w") as f:
            f.write(xml_str)
    return xml_str


def read_xml(xml_str: str = None, path: str = None):
    """xml string or path -> (encoding, [TensorSpec])."""
    if path is not None:
        root = ET.parse(path).getroot()
    else:
        root = ET.fromstring(xml_str)
    return from_element(root)
