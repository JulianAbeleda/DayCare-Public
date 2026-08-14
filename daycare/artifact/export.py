"""Export dispatch: turn a finished .xpkg into whichever runtime forms
configs/runtime-targets.xml declares (see runtime_targets.py).

The user's ask this answers: "xpkg is just one form factor. it can also be a
gguf. so this option should be selectable." This module is the seam that
makes the target *selectable* rather than each caller hardcoding "run
emit_gguf" -- a registry (EXPORTERS) plus one entry point (export()) that
tries every declared target independently and reports per-target success or
failure, instead of a script that dies on the first problem.

Design choices worth being explicit about:

  * export() dispatches over ALL targets runtime_targets.read() returns, not
    just the ones marked status="export". native-xml is status="experimental"
    in the XML, but it still needs a result in the returned dict -- it is a
    real (trivial) answer to "which form do you want", just a no-op one.
    runtime_targets.exportable() is a separate, informational filter for
    callers who want to distinguish "this actually writes something new".

  * Each target is attempted in isolation. A broken gguf export must not
    prevent native-xml (or a future target) from succeeding -- so failures are
    *returned*, not raised, and preflighted before anything is written where
    that is possible (see can_export).

  * emit_gguf is imported lazily, inside the gguf exporter function, not at
    module top. As of this writing daycare/artifact/gguf_kv.py no longer pulls
    the vendored train-only tinygrad at import time (only its read_gguf()
    function does, lazily), so emit_gguf happens to import cleanly even
    without it -- but the lazy import here is deliberate and kept anyway: it
    means this dispatch layer stays inspectable and importable regardless of
    what emit_gguf.py imports in the future, on any machine, even one with no
    tinygrad and no GPU.
"""
from __future__ import annotations

import os
from typing import Callable

from . import gguf_kv, package
from . import runtime_targets as rt

# identity_gguf.py is deliberately NOT registered here. Its *input* is an
# already-built GGUF file, not a .xpkg -- it rewrites a base model's chat
# template to gate an identity turn -- and it shells out to an external
# llama.cpp checkout (LLAMA_CPP_ROOT) to do it. That makes it a
# post-processing step chained *after* the "gguf" exporter below
# (gguf_path = export(...)["gguf"]; then identity_gguf.build(base=gguf_path,
# ...)), not a peer target with its own entry in this registry. Forcing it in
# here would blur "export an xpkg to a runtime form" with "further mutate an
# already-exported GGUF for one specific serving behaviour".


def _export_native_xml(pkg_dir: str, out_dir: str) -> str:
    """native-xml is a no-op: the .xpkg already IS this artifact. Return its
    path unchanged -- do not copy it, there is nothing to convert."""
    return pkg_dir


def _export_hf(pkg_dir: str, out_dir: str) -> str:
    # Never actually reached: can_export() gates "hf" out via UNAVAILABLE
    # before export() calls into EXPORTERS. Kept as a clearly-failing stub
    # (rather than omitted) so EXPORTERS enumerates every target export()
    # might dispatch, and so calling it directly is also honest about not
    # working, not silently wrong.
    raise NotImplementedError(UNAVAILABLE["hf"])


def _export_gguf(pkg_dir: str, out_dir: str) -> str:
    from . import emit_gguf  # lazy -- see module docstring

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.gguf")
    emit_gguf.emit(pkg_dir, out_path)
    return out_path


EXPORTERS: dict[str, Callable[[str, str], str]] = {
    "native-xml": _export_native_xml,
    "hf": _export_hf,
    "gguf": _export_gguf,
}

# Targets that are declared (in runtime-targets.xml and/or EXPORTERS) but
# have no working implementation. can_export() reports these directly rather
# than letting export() find out by calling the exporter and catching
# whatever it raises -- "hf" should read as "not built yet", not as a stack
# trace that happens to say NotImplementedError.
UNAVAILABLE: dict[str, str] = {
    "hf": "hf export is not implemented (only native-xml and gguf are)",
}


def can_export(pkg_dir: str, target: str) -> tuple[bool, str]:
    """Preflight `target` against `pkg_dir` before any writing happens.

    Returns (True, "") if the target looks exportable, else (False, reason)
    with a reason meant to be shown to a human, not just logged.

    This is what catches the int8/f16 footgun: package.write_package defaults
    to quant="int8", but gguf_kv.GGML_FOR_QUANT only maps f16/f32, so a
    default-built .xpkg asked for a "gguf" export would otherwise fail via
    emit_gguf.emit -- previously a bare SystemExit, which is not even
    catchable by `except Exception` (see emit_gguf.UnsupportedQuantError).
    Checking the quant here means export() can report it as one clean
    per-target failure instead of a mid-export crash that takes the rest of
    the dispatch loop down with it.
    """
    if target in UNAVAILABLE:
        return False, UNAVAILABLE[target]
    if target not in EXPORTERS:
        return False, f"unknown target {target!r}; known: {sorted(EXPORTERS)}"

    if target == "native-xml":
        if not os.path.isdir(pkg_dir):
            return False, f"no package directory at {pkg_dir!r}"
        return True, ""

    if target == "gguf":
        try:
            manifest, _specs = package.read_index(pkg_dir)
        except Exception as exc:  # e.g. missing/corrupt manifest.xml
            return False, f"cannot read package at {pkg_dir!r}: {exc}"
        if manifest.quant not in gguf_kv.GGML_FOR_QUANT:
            return False, (
                f"quant={manifest.quant!r} has no GGUF mapping; supported: "
                f"{sorted(gguf_kv.GGML_FOR_QUANT)} (package.write_package "
                "defaults to quant=\"int8\", which is not one of them)"
            )
        return True, ""

    return True, ""  # future targets with no special preflight


def export(
    pkg_dir: str,
    out_dir: str,
    targets_xml: str,
    only: list[str] | None = None,
) -> dict[str, str | Exception]:
    """Export `pkg_dir` to every target declared in `targets_xml`.

    Returns {target_name: out_path} on success or {target_name: Exception}
    on failure, one entry per attempted target. `only`, if given, restricts
    the attempt to that subset of declared target names (unknown names in
    `only` are simply not attempted -- they never appear in the result).

    Each target is fully independent: a preflight failure or an exception
    raised mid-export is caught and recorded, and dispatch continues to the
    next target rather than aborting the whole call.
    """
    targets = rt.read(targets_xml)
    names = [t.name for t in targets]
    if only is not None:
        names = [n for n in names if n in only]

    results: dict[str, str | Exception] = {}
    for name in names:
        ok, reason = can_export(pkg_dir, name)
        if not ok:
            results[name] = RuntimeError(reason)
            continue
        try:
            results[name] = EXPORTERS[name](pkg_dir, out_dir)
        except Exception as exc:  # isolate: one bad target must not stop the rest
            results[name] = exc
    return results
