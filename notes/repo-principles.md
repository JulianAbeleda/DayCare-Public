# DayCare Repo Principles

DayCare is a **development repo**. It holds both implementation code (primarily
the model agent) and the research notebook that informs it.

Rules:

- Code and Markdown both belong here. Implementation lives under the `daycare/`
  package; research notes live under `research/`.
- Research notes: prefer primary papers and official docs; cite every non-obvious
  claim with a source URL; keep summaries opinionated but separable from facts.
- Keep `research/` as the design record. Code should trace back to a research
  note when it implements a non-trivial design decision.
- Do not commit datasets, model weights, benchmark outputs, or training runs.
  Persistent agent state (e.g. a model's `body.json`) is runtime data, not source
  — keep real instances out of git; commit only small example/fixture bodies.
- Keep dependencies explicit and minimal; pin what matters for reproducibility.
