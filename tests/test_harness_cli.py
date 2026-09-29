"""`python -m daycare.harness` runs from the published tree (it once imported an unpublished module)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_help_runs():
    done = subprocess.run([sys.executable, '-m', 'daycare.harness', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert '--workload' in done.stdout


def test_simulated_run_emits_a_ledger(tmp_path, monkeypatch):
    from daycare.app import run as run_mod
    from daycare.harness.__main__ import main
    monkeypatch.setattr(run_mod._SimulatedBackend, '_STEP_DELAY_S', 0.0)
    out = tmp_path / 'ledger.xml'
    code = main(['--workload', 'repo idiom', '--out', str(out)])
    text = out.read_text()
    assert '<run-ledger' in text and 'complete at step 2000/2000' in text
    assert code == 1  # simulated authority never promotes
