# Branch layout: `main` ⊂ `dev` ⊂ `exp`

Work begins on `exp`. What proves useful is reduced to reusable development
apparatus on `dev` and the smallest durable product or research conclusion on
`main`. Promotion is curation: experiments and their scaffolding do not move
inward wholesale.

DayCare follows the same three-branch convention as BoltBeam and
tinygrad-arkey. The relationship is **containment by content category**, not a
promotion pipeline. Work does not graduate from `exp` to `dev` to `main`;
rather, each branch admits categories of content that the one inside it refuses.

## What each branch admits

### `main` — the runnable product surface

What the project *is* when someone runs it:

- the `daycare/` package, the CLI, the model adapters
- `examples/` — runnable example artifacts and scaffolds
- `research/` — the knowledge base. A record of what you learned is product:
  findings, scopes, maps, and the bibliography live here
- `notes/`, `README.md`, `.gitignore`
- `docs/branch-flow.md` and `tools/sync_branches.sh` — the maintenance apparatus
  that keeps this layout, mirroring BoltBeam

Excluded by category, regardless of quality:

- **the test suite** — see below
- one-off scripts, probes, and campaign tooling
- prototype interfaces and exploratory builds
- runtime state, model weights, and training outputs (not in git at all)

### Why the tests are not on `main`

A test's purpose is to *establish that the product is correct*. That is
producing knowledge about the product, not being the product. `dev` is defined
as "main plus what you need to verify it"; the suite is precisely that. The
trunk's first verification is the merge into `dev`, which is hard-gated on the
suite. A trunk commit that breaks the product cannot reach `exp` without
failing that gate.

### `dev` — main plus the apparatus

Everything on `main`, plus what you need to *verify* and *develop* it:

- **the test suite** — `tests/`; `pytest` is run from here
- development and diagnostic tooling — `scripts/` (e.g. the mermaid rendering
  helpers)
- durable handoffs and finished investigation records that are not product

### `exp` — active experimentation

Prototypes and probes in flight:

- `gui/` — interface explorations and prototype views
- machine-search and hardware experiments

Broken intermediate states are fine here; the suite need not pass. Most of what
lands here should never move inward. Its value is the answer it produces, not
the code. Once a probe's verdict is recorded, the record moves to `research/`
on `main` and the probe can go.

## The admission test

Before adding a file to `main`, ask which of these it is:

| If it is… | It belongs on |
| --- | --- |
| part of what the tool does when run | `main` |
| how you check that the tool is correct | `dev` |
| how you found out what to build | `exp` |
| a record of what you learned | `research/` on `main` |

## Keeping them in sync

Because the relationship is containment, **`main` flows outward** — every
change to the product surface must reach `dev` and `exp`, or they stop being
supersets and start being forks:

```
./tools/sync_branches.sh --dry-run   # report what each merge would delete
./tools/sync_branches.sh             # merge outward, suite-gated at each hop
git push origin dev exp
```

Do this on every trunk commit, or close to it.

### The prune case

One situation needs a human. When the trunk **prunes** something an outer
branch retains, an ordinary merge deletes it there too — the exact opposite of
what the layout is for.

The mechanism is an **apparatus commit**: after the prune merges outward,
restore the retained paths on the outer branch and commit them. `dev` is then
"trunk plus apparatus" by construction, and the restoration sits above the
prune in history, so later merges do not re-delete it.

```
git checkout dev
git merge main                              # brings the prune
git checkout <pre-prune-sha> -- <paths>     # put back what dev retains
git commit -m "[repo] retain <what> as dev apparatus"
```

**Restore the constraining tests too, not just the code.** An apparatus layer
that restores code without the test that constrains it leaves the branch
self-inconsistent, and the sync script fails loudly on exactly that.

Movement inward is not a merge. Promoting something from `exp` to `main` means
deciding it was product all along; it lands as an ordinary commit on `main`
that then flows outward like any other.

## Direction matters

The human workflow moves from experiment toward survival:

```
exp --curate--> dev --curate--> main
```

Git synchronization moves in the opposite direction so the outer branches stay
supersets:

```
main --merge--> dev --merge--> exp
```

This is a rule about **what kind of thing** each branch holds. It keeps the
trunk lean while preserving the apparatus and experiments needed to explain it.

Run `python3 sz.py` on every branch to keep those surfaces visible. The durable
product budget applies only to runnable code under `daycare/` and `examples/`;
tests, tooling, experiments, and documentation are reported and bounded
separately so moving code cannot hide growth.
