"""GameTerm's model-facing tool-result content, the one DayCare copy.

GameTerm owns this contract (`crates/provider/src/lib.rs`, `WireToolResult`,
serialized with `serde_json::to_string`): a compact JSON object whose field
order is transport, stdout, stderr, exit_code, truncated. `capability_outcome`
and `effect` are omitted when absent, which is the case for `calculate`.
Training rows and probes must show the model these bytes, not a bare receipt,
or training and serving disagree about what a tool result looks like.
"""
import json
import subprocess

TRANSPORTS = ("succeeded", "failed", "unknown", "rejected", "canceled")


def tool_content(transport: str, stdout: str = "", stderr: str = "", *,
                 exit_code: int | None = None, truncated: bool = False) -> str:
    if transport not in TRANSPORTS:
        raise ValueError(f"unknown GameTerm transport outcome: {transport}")
    return json.dumps({"transport": transport, "stdout": stdout, "stderr": stderr,
                       "exit_code": exit_code, "truncated": truncated},
                      separators=(",", ":"), ensure_ascii=False)


def calculator_content(outcome: str) -> str:
    """A successful `calculate` call; `outcome` is the runner's `expr = value`."""
    return tool_content("succeeded", stdout=outcome)


def rejected_content(message: str) -> str:
    """A call GameTerm refused before running it; the reason is on stderr."""
    return tool_content("rejected", stderr=message)


def calculate(runner, expression: str) -> str:
    """The real GameTerm calculate runner's `expr = value` receipt; refuses a failed call."""
    receipt = json.loads(subprocess.run([str(runner)], input=json.dumps({"expression": expression}) + "\n",
                                        text=True, capture_output=True, timeout=30, check=True).stdout)
    if not receipt["ok"]:
        raise ValueError(receipt)
    return receipt["outcome"]


def exchange(expression: str, outcome, ident: str, content: str | None = None) -> list[dict]:
    """One calculate call and its result exactly as GameTerm serves them."""
    return [{"role": "assistant", "content": None, "tool_calls": [{
        "id": ident, "type": "function", "function": {
            "name": "calculate", "arguments": {"expression": expression}}}]},
        {"role": "tool", "tool_call_id": ident, "content": content or calculator_content(outcome)}]
