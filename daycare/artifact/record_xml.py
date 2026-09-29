"""Lossless, typed XML records for experiment metadata."""
from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")  # not allowed in XML 1.0 text


def _encode(parent: ET.Element, value) -> None:
    if value is None:
        parent.set("type", "null")
    elif isinstance(value, bool):
        parent.set("type", "bool")
        parent.text = "true" if value else "false"
    elif isinstance(value, int):
        parent.set("type", "int")
        parent.text = str(value)
    elif isinstance(value, float):
        parent.set("type", "float")
        parent.text = repr(float(value))  # np.float64 subclasses float; its NumPy 2 repr is "np.float64(x)"
    elif isinstance(value, str) and ILLEGAL.search(value):  # e.g. a sampled "\b" (0x08): kept as JSON, lossless
        parent.set("type", "json_str")
        parent.text = json.dumps(value)
    elif isinstance(value, str):
        parent.set("type", "str")
        parent.text = value
    elif isinstance(value, dict):
        parent.set("type", "dict")
        for key, item in value.items():
            child = ET.SubElement(parent, "field", name=str(key))
            _encode(child, item)
    elif isinstance(value, (list, tuple)):
        parent.set("type", "list")
        for item in value:
            _encode(ET.SubElement(parent, "item"), item)
    else:
        raise TypeError(f"XML records do not support {type(value).__name__}")


def _decode(node: ET.Element):
    kind = node.get("type")
    if kind == "null":
        return None
    if kind == "bool":
        return node.text == "true"
    if kind == "int":
        return int(node.text or "0")
    if kind == "float":
        return float(node.text or "0")
    if kind == "str":
        return node.text or ""
    if kind == "json_str":
        return json.loads(node.text)
    if kind == "dict":
        return {child.get("name"): _decode(child) for child in node.findall("field")}
    if kind == "list":
        return [_decode(child) for child in node.findall("item")]
    raise ValueError(f"unknown XML record type: {kind!r}")


def write(path: str | Path, value, *, root: str = "record") -> None:
    path = Path(path)
    if path.suffix != ".xml":
        raise ValueError("experiment records must use an .xml filename")
    document = ET.Element(root, schema="daycare.record.v1")
    _encode(document, value)
    ET.indent(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)


def read(path: str | Path):
    root = ET.parse(path).getroot()
    if root.get("schema") != "daycare.record.v1":
        raise ValueError("unsupported XML record schema")
    return _decode(root)
