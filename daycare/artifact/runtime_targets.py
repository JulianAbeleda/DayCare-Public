"""Read configs/runtime-targets.xml -- the declared runtime forms an artifact
can be served as (xml.etree, stdlib only; mirrors the style of manifest.py).

Today that file is pure documentation (see
examples/xml-pretraining-repo/configs/runtime-targets.xml): three
`<target name=... status=.../>` elements and nothing that reads them. This
module is what makes it machine-readable, so daycare/artifact/export.py can
dispatch off the XML instead of a hardcoded list living in Python.

`status` is the XML author's own claim about a target:
  - "export"       -- a genuinely new artifact form (hf, gguf)
  - "experimental" -- native-xml: the .xpkg IS the artifact already: shipping
    it *as* a target is documentation of that fact, not a new form to build.

Note the XML's claim and the code's reality can diverge -- it currently
declares "hf" as status="export" even though no exporter exists yet
(daycare/artifact/export.py registers it as explicitly unavailable). This
module only reports what the XML says; export.py is what reconciles that
against what is actually implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class RuntimeTarget:
    name: str
    status: str


def read(path: str) -> list[RuntimeTarget]:
    """runtime-targets.xml -> [RuntimeTarget], in document order."""
    root = ET.parse(path).getroot()
    if root.tag != "runtime-targets":
        raise ValueError(f"unknown root tag: {root.tag!r} (expected 'runtime-targets')")
    return [RuntimeTarget(name=el.get("name"), status=el.get("status", ""))
            for el in root.findall("target")]


def exportable(targets: list[RuntimeTarget]) -> list[RuntimeTarget]:
    """Targets the XML itself claims are status="export" -- i.e. a genuinely
    new artifact form, as opposed to native-xml's "experimental" pass-through.

    This is informational (what the XML declares), not a gate on what
    daycare/artifact/export.py will attempt -- export() also dispatches
    native-xml, since returning the xpkg path unchanged is still a valid
    (trivial) answer to "what form do you want this artifact in".
    """
    return [t for t in targets if t.status == "export"]


def by_name(targets: list[RuntimeTarget], name: str) -> RuntimeTarget | None:
    return next((t for t in targets if t.name == name), None)
