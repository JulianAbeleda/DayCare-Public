"""Read/write <adapter> and <model> manifests (xml.etree, stdlib only).

Boundary: this module owns the <adapter>/<model> wrapper element and its
provenance/config children. It treats the <tensors> payload block as an
OPAQUE ET.Element -- it neither parses nor validates its contents (that's
daycare/artifact/index.py's job). On write, pass tensors_el to embed it
as-is; on read, the raw <tensors> child is exposed unparsed via
`tensors_el` on AdapterManifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

@dataclass
class AdapterManifest:
    subject: str
    base: str
    r: int
    alpha: float
    last_k: int
    targets: list[str]
    epochs: int
    loss_start: float
    loss_end: float
    eval_metric: str
    eval_before: str
    eval_after: str
    eval_gate: str
    method: str = "lora"
    source_sha256: str = ""
    architecture: str = ""
    factors: list[str] = field(default_factory=list)
    target_map: list[dict] = field(default_factory=list)
    tensors_el: ET.Element | None = None

    def to_xml(self) -> ET.Element:
        root = ET.Element(
            "adapter", subject=self.subject, base=self.base, method=self.method
        )
        ET.SubElement(
            root,
            "lora",
            r=str(self.r),
            alpha=str(self.alpha),
            **{"last-k": str(self.last_k)},
            targets=",".join(self.targets),
        )
        ET.SubElement(
            root,
            "source",
            sha256=self.source_sha256,
            architecture=self.architecture,
        )
        factors = ET.SubElement(root, "factors")
        for name in self.factors:
            ET.SubElement(factors, "factor", name=name)
        targets = ET.SubElement(root, "target-map")
        for item in self.target_map:
            shape = item.get("shape", ())
            ET.SubElement(
                targets,
                "target",
                adapter=str(item.get("index", "")),
                tensor=str(item.get("tensor", "")),
                shape="x".join(str(v) for v in shape),
                layer=str(item.get("layer", "")),
                name=str(item.get("target", "")),
            )
        training = ET.SubElement(
            root,
            "training",
            epochs=str(self.epochs),
            **{"loss-start": str(self.loss_start), "loss-end": str(self.loss_end)},
        )
        ET.SubElement(
            training,
            "eval",
            metric=self.eval_metric,
            before=self.eval_before,
            after=self.eval_after,
            gate=self.eval_gate,
        )
        if self.tensors_el is not None:
            root.append(self.tensors_el)
        return root

    @staticmethod
    def from_xml(el: ET.Element) -> "AdapterManifest":
        lora = el.find("lora")
        training = el.find("training")
        ev = training.find("eval")
        tensors_el = el.find("tensors")
        source = el.find("source")
        factor_el = el.find("factors")
        target_el = el.find("target-map")
        target_map = []
        if target_el is not None:
            for node in target_el.findall("target"):
                shape = tuple(int(v) for v in node.get("shape", "").split("x") if v)
                target_map.append({
                    "index": int(node.get("adapter", "0")),
                    "tensor": node.get("tensor", ""),
                    "shape": shape,
                    "layer": int(node.get("layer", "0")),
                    "target": node.get("name", ""),
                })
        return AdapterManifest(
            subject=el.get("subject"),
            base=el.get("base"),
            method=el.get("method", "lora"),
            r=int(lora.get("r")),
            alpha=float(lora.get("alpha")),
            last_k=int(lora.get("last-k")),
            targets=lora.get("targets").split(","),
            epochs=int(training.get("epochs")),
            loss_start=float(training.get("loss-start")),
            loss_end=float(training.get("loss-end")),
            eval_metric=ev.get("metric"),
            eval_before=ev.get("before"),
            eval_after=ev.get("after"),
            eval_gate=ev.get("gate"),
            source_sha256=source.get("sha256", "") if source is not None else "",
            architecture=source.get("architecture", "") if source is not None else "",
            factors=[node.get("name", "") for node in factor_el.findall("factor")]
                    if factor_el is not None else [],
            target_map=target_map,
            tensors_el=tensors_el,
        )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ModelManifest:
    """<model> manifest.

    `kv` is the source model's key/value metadata as (key, type_name, text) --
    the *exact* original key (dots and all; XML attribute names allow them) and
    its type, so the artifact is self-describing and can be re-emitted to a GGUF
    without a lookup table. Values stay text here; daycare/artifact/gguf_kv.py
    converts text <-> typed value.
    """

    name: str
    arch: str
    quant: str
    kv: list[tuple[str, str, str]] = field(default_factory=list)
    tokenizer_ref: str = ""
    tensors_ref: str = ""
    source_sha256: str = ""
    weights_sha256: str = ""

    def to_xml(self) -> ET.Element:
        root = ET.Element("model", name=self.name, arch=self.arch, quant=self.quant)
        ET.SubElement(root, "source", sha256=self.source_sha256, weights_sha256=self.weights_sha256)
        config = ET.SubElement(root, "config")
        for key, typ, val in self.kv:
            el = ET.SubElement(config, "kv", key=key, type=typ)
            el.text = val
        ET.SubElement(root, "tokenizer", ref=self.tokenizer_ref)
        ET.SubElement(root, "tensors", ref=self.tensors_ref)
        return root

    @staticmethod
    def from_xml(el: ET.Element) -> "ModelManifest":
        config_el = el.find("config")
        tokenizer_el = el.find("tokenizer")
        tensors_el = el.find("tensors")
        kv = []
        if config_el is not None:
            for e in config_el.findall("kv"):
                kv.append((e.get("key"), e.get("type"), e.text or ""))
        return ModelManifest(
            name=el.get("name"),
            arch=el.get("arch"),
            quant=el.get("quant"),
            kv=kv,
            tokenizer_ref=tokenizer_el.get("ref") if tokenizer_el is not None else "",
            tensors_ref=tensors_el.get("ref") if tensors_el is not None else "",
            source_sha256=el.find("source").get("sha256", "") if el.find("source") is not None else "",
            weights_sha256=el.find("source").get("weights_sha256", "") if el.find("source") is not None else "",
        )

    def get(self, key: str, default=None) -> str | None:
        """Look up a raw KV text value by its exact GGUF key."""
        for k, _t, v in self.kv:
            if k == key:
                return v
        return default


Manifest = AdapterManifest | ModelManifest


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def write(manifest: Manifest, path: str, tensors_el: ET.Element | None = None) -> None:
    if tensors_el is not None and isinstance(manifest, AdapterManifest):
        manifest.tensors_el = tensors_el
    root = manifest.to_xml()
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def read(path: str) -> Manifest:
    root = ET.parse(path).getroot()
    if root.tag == "adapter":
        return AdapterManifest.from_xml(root)
    if root.tag == "model":
        return ModelManifest.from_xml(root)
    raise ValueError(f"unknown manifest root tag: {root.tag}")
