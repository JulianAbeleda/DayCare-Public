"""DAYCARE_MODE: the one switch between an offline demo and the real thing.

`daycare/harness/verdict.py` refuses to promote a `simulated` run no matter
how good its numbers look, because a ledger that cannot tell you whether it
is real is a ledger nobody can trust. The interface owes the viewer the same
honesty about *itself*: a demo that cannot say whether it is live is the same
failure. So there is exactly one environment variable, read in exactly one
place, and everything downstream (the judge, the orchestrator, the badge in
the shell) asks this module rather than `os.environ` directly -- one source
means the judge and the shell can never quietly disagree about which mode is
running.

DEBUG is the default deliberately: a fresh clone, with no `claude` on PATH,
no ssh key, and no network, must still run the whole product. An unset or
misspelled `DAYCARE_MODE` (a typo, a shell that dropped the export) has to
fail toward that safe state rather than toward reaching out to a teacher,
an orchestrator, or a GPU host that may not exist on this machine -- so an
unknown value falls back to DEBUG rather than raising. Raising would only
turn a typo into a crash; falling back to LIVE would turn a typo into an
unwanted network call.
"""
from __future__ import annotations

import os
import threading

DEBUG = "debug"
LIVE = "live"

_ENV_VAR = "DAYCARE_MODE"
_VALID = frozenset({DEBUG, LIVE})

#: Set by the running app when someone flips the switch in the interface.
#: The environment names the mode the process STARTED in; this names the mode
#: it is in NOW. Demoing means moving between them mid-conversation, and
#: restarting the server to do that would make the difference feel like a
#: deployment rather than a choice.
_override: str | None = None
_lock = threading.Lock()


def current() -> str:
    """The active mode: the runtime override if one is set, else
    `DAYCARE_MODE`, defaulting to DEBUG.

    Fails closed: anything other than exactly "debug" or "live" -- unset,
    empty, mis-cased, misspelled -- resolves to DEBUG rather than raising.
    A typo must not silently arm the live path (network, the `claude` CLI,
    the GPU host); the worst a bad value can do is fall back to the mode
    that needs nothing external.
    """
    with _lock:
        if _override is not None:
            return _override
    raw = os.environ.get(_ENV_VAR, DEBUG).strip().lower()
    return raw if raw in _VALID else DEBUG


def set_mode(value: str) -> str:
    """Flip the switch at runtime; returns the mode now in force.

    Validated the same way `current()` validates the environment, and for the
    same reason -- a bad value from a client must land on DEBUG, not raise and
    not arm the live path. Process-wide by design: one server, one answer, so
    the badge in the browser and the judge the next request reaches for can
    never disagree about which mode is running.
    """
    global _override
    raw = (value or "").strip().lower()
    with _lock:
        _override = raw if raw in _VALID else DEBUG
        return _override


def clear_override() -> None:
    """Fall back to the environment. Mainly so tests cannot leak a mode into
    each other -- a stuck override would make an offline test quietly live."""
    global _override
    with _lock:
        _override = None


def is_live() -> bool:
    return current() == LIVE
