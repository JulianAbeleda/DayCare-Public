"""Provenance for training records: file digests and the exact DayCare revision a run used."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> str:
    """HEAD of this checkout; refuses uncommitted tracked changes, so a record names the code that ran."""
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"]).decode().strip():
        raise ValueError("commit tracked DayCare changes before training")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
