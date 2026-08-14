"""The single declarative description of what DayCare can do.

Before this module, "what DayCare can do" existed in four places that had
to be edited together and never were checked against each other:

  1. `daycare/orchestrator/__init__.py`'s `TOOL_NAMES` -- a tuple the parser
     used to decide whether a tool call names something real.
  2. That same module's `REQUIRED_ARGS` -- a dict deciding whether a call is
     complete enough to act on.
  3. `_SYSTEM_PROMPT`'s hand-written prose describing each action to the
     model, in its own words, independently of (1) and (2).
  4. `daycare/app/server.py`'s `_ROUTES` table -- what the app actually
     exposes over HTTP, a fourth list nothing here ever compared against
     the other three.

Add a route and forget any one of these and the model's idea of what DayCare
can do, the parser's idea, and what the server actually serves quietly
diverge. This module is the fix: one `Tool` per capability, and everything
downstream -- the orchestrator's system prompt, its parser's `TOOL_NAMES`/
`REQUIRED_ARGS`-shaped needs, and `GET /api/tools` in server.py -- derives
its answer from `TOOLS` rather than keeping a second copy.

Two axes distinguish entries, and both are load-bearing:

  `write` -- does calling this spend a GPU, change a model, or cost money?
  `start_run` occupies a GPU for minutes and can degrade a model; a bad
  `read_ledger` call, at worst, shows nobody anything. Downstream (a future
  approval step, the CLI) asks this flag rather than hardcoding a list of
  "the dangerous ones" that will fall out of date the next time someone
  adds a tool here and forgets to update that list too.

  `chat` -- may the conversational orchestrator emit this as a `<tool-call>`
  during `/api/chat`? `route`, `start_run`, `read_ledger`, and `artifacts`
  are what a person's *words* can trigger. Starting or stopping a
  llama-server and flipping DEBUG/LIVE are administrative -- the CLI (or a
  browser control) calls those routes directly, the same way `daycare
  servers` and `daycare mode` are their own top-level commands in
  `cli-as-a-harness-client.md`'s design, never something typed into a
  conversation. `chat=False` entries still belong in this registry (and
  still get served over `GET /api/tools`) because the registry's job is
  "everything DayCare can do," not just "everything the chat model can do"
  -- but the orchestrator must filter on this flag before it ever shows one
  to the model or accepts one back.

Only routes that already exist in `daycare/app/server.py` are listed here.
This module does not get to invent a capability the server doesn't serve.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arg:
    """One argument a tool call carries.

    `meaning` is prose -- what the value holds, in plain words, the same
    thing the old `_SYSTEM_PROMPT` said inline ("arg \"target\": the thing
    to teach, in the person's own words"). It is never itself an example
    value -- see `Example` below for where real values live.
    """
    name: str
    required: bool
    meaning: str


@dataclass(frozen=True)
class Example:
    """One worked example: something a person actually said, the sentence
    the orchestrator says back before acting, and the REAL argument values
    the resulting `<tool-call>` carries.

    This is the fix for a bug that already happened once: an earlier prompt
    showed the tool-call format as `<arg key="target">THE TARGET</arg>` --
    a placeholder standing in for "whatever the person said" -- and the
    live model dutifully copied `THE TARGET` verbatim into a real call. It
    can reproduce the format perfectly; it just reproduces exactly what it
    is shown, so what it is shown has to already be the finished article.
    `args` here must therefore always be values a person could plausibly
    have said, never a name that describes an argument rather than filling
    one in.
    """
    person: str
    says: str
    args: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    """One capability. `endpoint` is the HTTP route in `server.py`'s
    `_ROUTES` table that actually does the work -- named here so a client
    reading `GET /api/tools` learns not just that a capability exists but
    which wire call performs it, rather than having to keep its own map
    from tool name to route (the exact second-copy problem this module
    exists to close, just one layer further out)."""
    name: str
    purpose: str
    args: tuple[Arg, ...]
    write: bool
    chat: bool
    endpoint: tuple[str, str]  # (HTTP method, path pattern)
    examples: tuple[Example, ...] = ()
    #: Optional behavioural instruction -- WHEN to call this, not what it
    #: does. Only `route` and `start_run` carry one; `read_ledger` and
    #: `artifacts` need no timing rule beyond "when asked." Kept in the
    #: registry (rather than hand-written in the orchestrator's prompt, as
    #: it used to be) for the same reason `purpose` and `args` are: one
    #: place, so the prompt-builder has nothing left to say about a tool
    #: that isn't sourced from here.
    when: str = ""

    @property
    def required_args(self) -> tuple[str, ...]:
        """The argument this tool cannot safely run without -- same
        judgement `REQUIRED_ARGS` used to encode by hand: a `start_run`
        missing `concept` is not a run with a blank name, it is a run that
        should not exist."""
        return tuple(a.name for a in self.args if a.required)


#: Every capability DayCare exposes, in the order a client should think
#: about them. Order matters here for one visible reason: it is also the
#: order the orchestrator's system prompt lists actions in, so putting
#: `route` first keeps "decide before you promise anything" the first
#: thing the model reads, matching how it already had to behave.
TOOLS: tuple[Tool, ...] = (
    Tool(
        name="route",
        purpose=("Decide whether a target belongs in a file, retrieval, or "
                  "the model's weights, before anything is promised."),
        args=(
            Arg("target", True, "the thing to teach, in the person's own words"),
        ),
        write=False,
        chat=True,
        endpoint=("POST", "/api/route"),
        when=("Call this as soon as someone states what they want the model "
              "to learn -- before promising to train anything."),
        examples=(
            Example(
                person="teach it to write code the way my repo does",
                says="Let me work out where that belongs.",
                args={"target": "write code the way my repo does"},
            ),
            Example(
                person="call the model Mamser",
                says="Let me check how a name should be taught.",
                args={"target": "call the model Mamser"},
            ),
        ),
    ),
    Tool(
        name="start_run",
        purpose="Launch a training run against a target that has already been routed to the weights.",
        args=(
            Arg("concept", True, "the routed target"),
            Arg("plan", False, "how much of the corpus to train on: skim, balanced, or whole"),
        ),
        write=True,  # occupies a GPU for minutes and can degrade a model
        chat=True,
        endpoint=("POST", "/api/train/start"),
        when=("Only call this after a route call has said the target is a "
              "real weight target and the person has agreed to run it."),
        examples=(
            Example(
                person="yes, run the balanced plan",
                says="Starting it now -- watch the ledger.",
                args={"concept": "write code the way my repo does", "plan": "balanced"},
            ),
        ),
    ),
    Tool(
        name="read_ledger",
        purpose="Check a run's progress so far.",
        args=(
            Arg("run_id", True, "which run to check"),
        ),
        write=False,
        chat=True,
        endpoint=("GET", "/api/train/<id>/ledger"),
    ),
    Tool(
        name="artifacts",
        purpose="Fetch what a finished run produced.",
        args=(
            Arg("run_id", True, "which run to fetch artifacts for"),
        ),
        write=False,
        chat=True,
        endpoint=("GET", "/api/train/<id>/artifacts"),
    ),
    Tool(
        name="servers_start",
        purpose="Launch a named llama-server (substrate or orchestrator) on the GPU host.",
        args=(
            Arg("name", True, "which server: substrate or orchestrator"),
            Arg("evict", False, 'stop every OTHER llama-server on the box first: "true" or "false"'),
        ),
        write=True,  # loads a model onto the GPU and holds it there
        chat=False,  # administrative: a CLI/browser action, never a conversational one
        endpoint=("POST", "/api/servers/<name>/start"),
    ),
    Tool(
        name="servers_stop",
        purpose="Stop a named llama-server on the GPU host.",
        args=(
            Arg("name", True, "which server: substrate or orchestrator"),
        ),
        # Stopping a server that is serving someone else's run is a real
        # cost too (see server.py's `_servers_xml` docstring) -- this is
        # not the harmless inverse of servers_start.
        write=True,
        chat=False,
        endpoint=("POST", "/api/servers/<name>/stop"),
    ),
    Tool(
        name="mode",
        purpose="Switch between debug (offline, scripted) and live (the real teacher and orchestrator).",
        args=(
            Arg("to", True, "debug or live"),
        ),
        write=False,  # flips a flag; spends nothing by itself
        chat=False,
        endpoint=("PUT", "/api/mode"),
    ),
)


def by_name(name: str) -> Tool | None:
    """`None` rather than a `KeyError` -- callers (the orchestrator's
    parser chief among them) treat "not a real tool" as a normal outcome to
    fail closed on, not an exceptional one."""
    for t in TOOLS:
        if t.name == name:
            return t
    return None


def chat_tools() -> tuple[Tool, ...]:
    """The subset the conversational orchestrator may emit and accept --
    see the module docstring's `chat` axis. Everything the orchestrator's
    `TOOL_NAMES`/`REQUIRED_ARGS` used to hand-list is exactly this."""
    return tuple(t for t in TOOLS if t.chat)
