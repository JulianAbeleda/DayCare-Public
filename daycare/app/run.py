"""The training-run seam: start, poll, stream, and cancel a run.

Training is SIMULATED today (`_SimulatedBackend`, below): it replays the
training-console prototype's idle loss-curve animation as real, timed events instead of
CSS-frame updates. Later a `_RemoteBackend` will drive the real loop --
`daycare/nursery/lesson.py`'s shape (she answers -> score -> update ->
fade, gated every few turns by a forgetting probe) -- on a remote CUDA
host, translating its telemetry into the same `Run._emit(...)` calls.

That is the whole point of this file: `start`/`get`/`events`/`cancel` and
the `Run`/event shapes are the stable interface the rest of the app (and
the browser) is written against. Swapping the backend is a change to one
class, not a rewrite -- see `_Backend` below, and the `# TODO(real-data):`
note on `_SimulatedBackend`.

Event dicts: {"seq": int, "type": "step"|"eval"|"done"|"error",
              "step": int, "loss": float, "msg": str}
"""
from __future__ import annotations

import math
import os
import random
import threading
import time
import uuid

from . import mode

# --------------------------------------------------------------------------
# Run: identity, status, and an append-only, thread-safe event log.
# --------------------------------------------------------------------------

_TERMINAL = frozenset({"done", "cancelled", "failed"})


class Run:
    """One training run.

    `status`/`step`/`max_steps`/`started_at`/`id` are read directly by
    callers (`run.status`, etc.) -- CPython's GIL makes those single-attribute
    reads/writes atomic, so that's safe without extra ceremony. Anything that
    needs *coordination* -- appending an event, flipping status, waiting for
    the next one -- goes through `_lock`/`_cv` so concurrent HTTP handler
    threads (start/get/events/cancel can all fire at once) never see a torn
    event log or miss a wakeup.
    """

    def __init__(self, run_id: str, max_steps: int):
        self.id = run_id
        self.status = "queued"  # queued -> running -> done | cancelled | failed
        self.authority = "simulated"  # set from the backend by start()
        self.step = 0
        self.max_steps = max_steps
        self.started_at = time.time()
        self.last_loss = float("nan")

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._events: list[dict] = []
        self._cancel_requested = False
        self._thread: threading.Thread | None = None

    # -- mutators: only ever called from the run's own background thread --

    def _set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
            self._cv.notify_all()

    def _emit(self, kind: str, *, step: int, loss: float, msg: str = "") -> dict:
        """Append an event with the next sequence number and wake any waiters."""
        with self._lock:
            self.step = step
            self.last_loss = loss
            seq = len(self._events) + 1
            ev = {"seq": seq, "type": kind, "step": step, "loss": loss, "msg": msg}
            self._events.append(ev)
            self._cv.notify_all()
            return ev

    def snapshot(self) -> list[dict]:
        """Every event landed so far, as a copy, without blocking.

        `events()` is the live view: it waits for the next entry and is the
        right thing for a stream. A ledger GET wants the opposite -- whatever
        exists at this instant, returned immediately -- so it gets its own
        accessor rather than reaching past the lock into the log.
        """
        with self._lock:
            return list(self._events)

    # -- cancellation: cooperative, checked at the run loop's next step --

    @property
    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def request_cancel(self) -> None:
        """Flag a cancel. Real GPU training can't be preempted mid-step, so
        this does not stop anything itself -- it just asks the run loop to
        stop at its next step boundary, which is the honest amount of control
        anyone has over a training step already in flight (simulated or not).
        """
        with self._lock:
            self._cancel_requested = True
            self._cv.notify_all()


# --------------------------------------------------------------------------
# Backend interface: what actually drives a Run's steps forward.
# --------------------------------------------------------------------------

class _Backend:
    """What a training backend must do to plug into the Run seam.

    Every backend must declare an `authority`: what its numbers are allowed to
    be used for. A simulated run and a real one produce the same `<entry>`
    elements, so without this the ledger cannot tell you which it is looking
    at -- and a ledger you cannot attribute is not evidence. The vocabulary is
    the harness-principles one:

        simulated         invented telemetry. Diagnostic only, never a claim.
        local-diagnostic  a real run, but not the shipped path.
        whole-system      the real thing, end to end. The only promotable tier.

    `_SimulatedBackend` is the only implementation today. A `_RemoteBackend`
    would live right beside it: construct with the same `(values, recipes)`,
    POST them to the GPU host's own /train/start, then keep a connection to
    its telemetry (polling or its own SSE) and translate each line into the
    same `run._emit("step"|"eval", ...)` calls -- `lesson.py`'s
    `turn N kind r=... loss=...` prints, plus its periodic
    `[gate] turn N forgetting=H/T` line, are the real shape that telemetry
    will take. `start()`/`get()`/`events()`/`cancel()` and `Run` itself do
    not change; only the object built inside `start()` does.
    """

    authority = "simulated"

    def drive(self, run: Run) -> None:
        """Advance `run` to completion, emitting "step"/"eval" events as it
        goes, and return. Must check `run.cancel_requested` at each step
        boundary and return promptly (without raising) if it's set. Let
        exceptions escape for real failures -- the caller turns those into
        an "error" event and a "failed" status.
        """
        raise NotImplementedError


class _SimulatedBackend(_Backend):
    """Replays the training-console prototype's own idle loss-curve animation as real events.

    The curve -- `0.72 + 2.05 * exp(-3.4 * frac) + noise` -- is lifted
    verbatim from gui/training.html (exp branch)'s `frame()` (around line 850, `g.care`
    driving `run.step`/`run.loss`), so a run driven through this API traces
    the same shape the page's own animation already showed before any of
    this existed. The step increment (`max(1, round(max_steps / 260))`)
    mirrors that function's `run.step += Math.max(1, Math.round(+c.max/260))`
    too, so the two don't visibly disagree about pacing.

    # TODO(real-data): this whole class goes away in favor of a
    # `_RemoteBackend` once training runs on the GPU host -- see `_Backend`
    # above for exactly what replaces it.
    """

    _EVAL_PROBE = 8          # a stand-in for evaluate.PROBE's size
    _EVAL_EVERY_FRAMES = 12  # echoes lesson.py's DAYCARE_GATE_EVERY_TURNS default
    _STEP_DELAY_S = 0.03     # tens of ms/step, so the UI actually streams

    def __init__(self, values: dict, recipes: dict):
        self._values = values
        self._recipes = recipes

    def drive(self, run: Run) -> None:
        max_steps = max(1, run.max_steps)
        step_inc = max(1, round(max_steps / 260))
        frame = 0
        step = run.step
        while step < max_steps:
            if run.cancel_requested:
                return
            time.sleep(self._STEP_DELAY_S)
            if run.cancel_requested:
                return
            step = min(max_steps, step + step_inc)
            frac = step / max_steps
            loss = 0.72 + 2.05 * math.exp(-3.4 * frac) + (random.random() - .5) * 0.11
            frame += 1
            run._emit("step", step=step, loss=loss)
            if frame % self._EVAL_EVERY_FRAMES == 0 or step >= max_steps:
                hit = min(self._EVAL_PROBE, round(frac * self._EVAL_PROBE))
                run._emit(
                    "eval", step=step, loss=loss,
                    msg=f"forgetting-probe {hit}/{self._EVAL_PROBE}",
                )


# --------------------------------------------------------------------------
# Module-level registry + the contract's public functions.
# --------------------------------------------------------------------------

_DEFAULT_MAX_STEPS = 200

_runs: dict[str, Run] = {}
_runs_lock = threading.Lock()


def _max_steps_from(values: dict, recipes: dict) -> int:
    """Pull a step budget out of the submitted values, with fallbacks.

    The intended source is the schedule file's `max` field (the
    `sources.py` FILES shape: `values["schedule"]["max"]`, a string from an
    <input>). If that's missing or unparsable, fall back to the first
    `recipes["care"]` entry's `max`, and finally to a fixed default -- a run
    should always get *some* budget to animate toward rather than raising.
    """
    try:
        raw = (values or {}).get("schedule", {}).get("max")
        if raw is not None:
            return max(1, int(float(raw)))
    except (AttributeError, TypeError, ValueError):
        pass
    care = (recipes or {}).get("care")
    if isinstance(care, list) and care:
        first = care[0]
        if isinstance(first, dict) and "max" in first:
            try:
                return max(1, int(float(first["max"])))
            except (TypeError, ValueError):
                pass
    return _DEFAULT_MAX_STEPS


def _remote_revision(host: str) -> str:
    """What commit the GPU host is actually running.

    `_RemoteBackend` ships a curriculum over the wire and trusts the host to
    train it with whatever code happens to be checked out there -- but until
    now nothing recorded what that was. The first live run silently trained
    against a stale checkout, and nothing in the ledger would have revealed
    it; this is the fix. Provenance is part of a claim
    (`daycare/harness/artifact.py`'s `provenance()` does the same thing for
    the LOCAL revision), so this is its remote counterpart.

    Fails soft (`"unknown"`), not by raising: a provenance check that itself
    needs the network should not be why a run never gets to try.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host,
             "cd ~/DayCare && git rev-parse --short HEAD"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    rev = proc.stdout.strip()
    return rev if proc.returncode == 0 and rev else "unknown"


class _RemoteBackend(_Backend):
    """A real distillation run on the GPU host, driven over ssh.

    The teacher writes the curriculum here (that is where the operator's
    credentials live), it is shipped to the host, and
    `daycare.nursery.distill` trains against it there. Its `PROGRESS`
    lines on stderr become the same `step`/`eval` events the simulated
    backend emits, so nothing downstream changes.

    `whole-system` authority: this is the shipped path end to end -- laptop
    asks, teacher writes, GPU trains, gate rules -- not a stand-in for it.
    """

    authority = "whole-system"

    def __init__(self, values: dict, recipes: dict, *, concept: str,
                 host: str, epochs: int = 2, max_probe: int = 4,
                 models_dir: str = "~/models"):
        self._concept = concept
        self._host = host
        self._epochs = epochs
        self._max_probe = max_probe
        self._models = models_dir
        self.document: str = ""
        self.adapter_path: str = ""

    def _save_path(self, run_id: str) -> str:
        """Where a durable adapter for this run would land on the host.

        Derived from the run id (not the concept) so concurrent runs never
        collide and the path is recoverable from the ledger alone -- the same
        reasoning, just for the shape
        `daycare.artifact.save.write_adapter` writes (one `<adapter>`
        document, not a stem with two extensions).
        """
        return f"{self._models}/daycare-{run_id}.adapter.xml"

    def drive(self, run: Run) -> None:
        import subprocess
        import tempfile
        import xml.etree.ElementTree as ET

        from ..nursery.fixtures import judge_for_mode
        from ..nursery.fixtures import judge_for_mode
        from ..nursery.generate import generate
        from ..nursery.route import Router
        from ..nursery.score import score_probe

        judge = judge_for_mode(mode.current())

        run._emit("eval", step=0, loss=float("nan"),
                  msg=f"remote revision on {self._host}: {_remote_revision(self._host)}")

        # The gate guards the GPU, not just the button. server.py routes the
        # target to choose a mechanism, but this is the thing about to spend an
        # hour of a graphics card, and it should not take that on trust from its
        # caller. The first real distillation run taught "lead with the command"
        # -- which routes FILE at step 1 -- and degraded the model for it.
        verdict = Router(judge).classify(self._concept)
        if not verdict.trainable:
            raise RuntimeError(
                f"refusing to train {self._concept!r}: it routes to "
                f"{verdict.route} at step {verdict.decided_at}. {verdict.because}"
            )
        run._emit("eval", step=0, loss=float("nan"),
                  msg=f"gate: trainable (step {verdict.decided_at})")

        run._emit("eval", step=0, loss=float("nan"),
                  msg=f"teacher is writing a curriculum for: {self._concept}")
        curriculum = generate(self._concept, judge, routed=True)
        n_ex = len(curriculum.findall("ex"))
        run._emit("eval", step=0, loss=float("nan"),
                  msg=f"curriculum ready: {n_ex} examples")
        if run.cancel_requested:
            return

        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
            fh.write(ET.tostring(curriculum, encoding="unicode"))
            local = fh.name
        remote = f"/tmp/daycare-curriculum-{run.id}.xml"
        subprocess.run(["scp", "-q", local, f"{self._host}:{remote}"],
                       check=True, timeout=60)

        # Training knobs have to travel: ssh does not carry the caller's
        # environment, so a learning rate tuned here would silently be the
        # remote default there -- and the default is what turned a model into
        # the token "1." on its first real run.
        knobs = " ".join(
            f"{k}={v}" for k in ("DAYCARE_FW_LR", "DAYCARE_FW_DECAY",
                                 "DAYCARE_LORA_LAST_K")
            if (v := os.environ.get(k))
        )
        save_path = self._save_path(run.id)
        cmd = (f"cd ~/DayCare && {knobs} python3 -m daycare.nursery.distill {remote} "
               f"--epochs {self._epochs} --max-probe {self._max_probe} "
               f"--save {save_path}")
        proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", self._host, cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for line in proc.stderr:
                if run.cancel_requested:
                    proc.terminate()
                    return
                if not line.startswith("PROGRESS "):
                    continue
                f = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                phase = f.get("phase")
                if phase == "step":
                    run._emit("step", step=int(f.get("n", 0)),
                              loss=float(f.get("loss", "nan")))
                elif phase == "gate":
                    run._emit("eval", step=run.step, loss=float("nan"),
                              msg=f"forgetting-probe {f.get('hit')}/{f.get('tot')} "
                                  f"({f.get('when')})")
                elif phase == "loaded":
                    run._emit("eval", step=0, loss=float("nan"),
                              msg=f"trainer loaded on {self._host} in {f.get('seconds')}s")
            self.document = proc.stdout.read()
        finally:
            proc.wait(timeout=30)
        if proc.returncode:
            raise RuntimeError(f"distill exited {proc.returncode} on {self._host}")

        # `distill.py` emits its own <save> element only if `--save` was
        # passed (see that module's docstring): "requested"/"gate" always,
        # "path" only if the forgetting-probe gate actually passed. Reading
        # it back here is how this backend learns whether an adapter is
        # actually sitting on the host, rather than just knowing what it
        # asked for.
        distill_el = ET.fromstring(self.document)
        save_el = distill_el.find("save")
        if save_el is not None and save_el.get("path"):
            self.adapter_path = save_el.get("path", "")
        gate = save_el.get("gate", "n/a") if save_el is not None else "n/a"
        run._emit("eval", step=run.step, loss=float("nan"),
                  msg=f"adapter save: gate={gate} "
                      f"path={self.adapter_path or '(not saved)'}")

        # `distill.py` deliberately does not score its own probe -- the
        # student marking its own homework -- so that is done here, with the
        # same judge the rest of the mode picked (`judge_for_mode` never
        # reaches for the real teacher in debug, however this backend got
        # constructed).
        judge = judge_for_mode(mode.current())
        try:
            probe_score = score_probe(distill_el, self._concept, judge)
        except ValueError as exc:
            run._emit("eval", step=run.step, loss=float("nan"),
                      msg=f"probe not scored: {exc}")
        else:
            run._emit("eval", step=run.step, loss=float("nan"),
                      msg=ET.tostring(probe_score.xml, encoding="unicode"))
            run._emit("eval", step=run.step, loss=float("nan"),
                      msg=f"probe {'learned' if probe_score.learned else 'NOT learned'}: "
                          f"fire {probe_score.after.fire_hit}/{probe_score.after.fire_total} "
                          f"silent {probe_score.after.silent_hit}/{probe_score.after.silent_total}")



def start(values: dict, recipes: dict, *, concept: str = "") -> Run:
    """Create a Run, register it, and hand it to a background thread.

    Returns immediately -- the HTTP layer must never block on a run, and a
    run must survive its client disconnecting (the thread is daemonic only
    so it can't wedge process shutdown; it isn't relied on for cleanup,
    since it always exits on its own once the run reaches a terminal state).

    Which backend runs is decided by `daycare.app.mode`, not a bespoke env
    var: `DAYCARE_MODE=live` (default DEBUG) is the one on-screen switch the
    whole app reads, the same way `harness/verdict.py` refuses to promote a
    `simulated` run however good it looks -- a demo that cannot say whether
    it is live is the same failure. The default stays simulated: a backend
    that silently reached for a GPU would make every existing caller slow
    and every test networked.
    """
    max_steps = _max_steps_from(values, recipes)
    run = Run(uuid.uuid4().hex[:12], max_steps)
    live = mode.is_live()
    if live and concept:
        backend = _RemoteBackend(
            values, recipes, concept=concept,
            host=os.environ.get("DAYCARE_TRAIN_HOST", "linuxbox"),
            epochs=int(os.environ.get("DAYCARE_EPOCHS", "2")),
            max_probe=int(os.environ.get("DAYCARE_MAX_PROBE", "4")),
            models_dir=os.environ.get("DAYCARE_MODELS_DIR", "~/models"),
        )
        run.max_steps = 0  # the curriculum decides; not known until it is written
    else:
        backend = _SimulatedBackend(values, recipes)
    # Carried onto the run so the ledger can say what produced it. Anything
    # reading these numbers has to be able to tell invented telemetry from a
    # real GPU, and the only honest place to record that is at the source.
    run.authority = backend.authority
    # The server needs the artifacts a finished run produced (a remote run
    # leaves an adapter and a GGUF on the host). Keeping the backend on the run
    # is how those paths get out without every backend having to agree on a
    # shape they mostly do not share.
    run.backend = backend

    def _body() -> None:
        run._set_status("running")
        try:
            backend.drive(run)
        except Exception as exc:  # a real backend WILL fail sometimes (host asleep, OOM, ...)
            run._emit("error", step=run.step, loss=float("nan"), msg=str(exc))
            run._set_status("failed")
            return
        if run.cancel_requested:
            run._set_status("cancelled")
        else:
            total = f"{run.step}/{run.max_steps}" if run.max_steps else str(run.step)
            run._emit("done", step=run.step, loss=run.last_loss,
                       msg=f"complete at step {total}")
            run._set_status("done")

    thread = threading.Thread(target=_body, name=f"daycare-run-{run.id}", daemon=True)
    run._thread = thread
    with _runs_lock:
        _runs[run.id] = run
    thread.start()
    return run


def get(run_id: str) -> Run | None:
    """Look up a run by id, or None if it never existed (or the process restarted --
    runs are in-memory only, matching everything else being mock/local for now)."""
    with _runs_lock:
        return _runs.get(run_id)


def events(run_id: str, since: int = 0):
    """Yield event dicts for `run_id` with seq > `since`, then keep yielding
    new ones as the run progresses, stopping cleanly once the run reaches a
    terminal status and every event up to that point has been delivered.

    `since` is what lets a client reconnect after a dropped SSE connection
    (laptop sleep, tab reload) and resume exactly where it left off instead
    of re-seeing the whole run or missing the tail of it. If `run_id` is
    unknown this simply yields nothing -- a not-found stream just ends.
    """
    run = get(run_id)
    if run is None:
        return
    seq = since
    while True:
        with run._cv:
            run._cv.wait_for(
                lambda: len(run._events) > seq or run.status in _TERMINAL,
                timeout=1.0,
            )
            pending = run._events[seq:]
            status = run.status
        for ev in pending:
            yield ev
        seq += len(pending)
        if status in _TERMINAL and not pending:
            return


def cancel(run_id: str) -> None:
    """Ask a run to stop. Cooperative: see `Run.request_cancel`. A no-op if
    the run doesn't exist or has already finished."""
    run = get(run_id)
    if run is None:
        return
    run.request_cancel()
