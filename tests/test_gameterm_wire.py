import json

import pytest

from daycare.harness.gameterm_wire import calculator_content, rejected_content, tool_content

# Captured verbatim from a GameTerm calculator_study request-01.json (2026-09-24).
CAPTURED = ('{"transport":"succeeded","stdout":"(70 * 70) / (70 - 1) = 71.0144927536",'
            '"stderr":"","exit_code":null,"truncated":false}')


def test_calculator_content_matches_captured_gameterm_bytes():
    assert calculator_content("(70 * 70) / (70 - 1) = 71.0144927536") == CAPTURED


def test_rejection_reason_is_stderr_with_rejected_transport():
    content = json.loads(rejected_content("malformed arguments: see the tool schema"))
    assert content["transport"] == "rejected" and content["stdout"] == ""
    assert content["stderr"].startswith("malformed arguments")


def test_unknown_transport_is_refused():
    with pytest.raises(ValueError):
        tool_content("ok")
