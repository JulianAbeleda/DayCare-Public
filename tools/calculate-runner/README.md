# gameterm-calculate-runner

GameTerm's `calculate` tool as a headless runner: the calculator DayCare's post-tool RL episodes call
(`daycare/harness/posttool.py`, `docs/rl-training.md`). It is GameTerm's calculator code (argument normalisation,
evaluation with [fend](https://github.com/printfn/fend), the one-line outcome) with nothing else from GameTerm, so a
DayCare run sees the same result bytes GameTerm serves.

## Build

Needs Rust 1.88 (`rust-toolchain.toml` pins 1.88.0; fend-core 1.5.8 uses let-chains). `Cargo.lock` pins every crate.

```bash
cd tools/calculate-runner
cargo build --release --locked
cargo test --release --locked
export DAYCARE_CALCULATE_RUNNER=$PWD/target/release/gameterm-calculate-runner
```

## Protocol

No arguments; the process stays alive for the whole run. One call per **stdin** line (the `calculate` arguments as
one line of JSON), one JSON reply per **stdout** line:

| reply | meaning | what DayCare shows the model |
|---|---|---|
| `{"ok":true,"outcome":"12 / 5 = 2.4"}` | an answer | `succeeded`, outcome on stdout |
| `{"error":"arguments: MalformedArguments","ok":false}` | schema error | `rejected`, fixed text on stderr (below) |
| `{"error":"could not calculate `1/0`: division by zero. ...","ok":false}` | calculator refusal | `succeeded`, text on stdout |

Keys are serialised in sorted order (`serde_json` without `preserve_order`), so an error reply starts with `error`.

## Arguments

`{"expression": string, "decimals"?: integer}`, nothing else.

- Any other field, a wrong type, or invalid JSON: `MalformedArguments` (so `{"expr": ...}` is malformed, not missing).
- No `expression` (or `null`): `MissingField`.
- `expression` has its whitespace runs collapsed to one space and is trimmed; empty or over 300 characters:
  `OutOfRange`. `decimals` outside 0..=12: `OutOfRange`.
- `round(X, N)` as the whole expression is `X` at `N` decimals (an explicit `decimals` wins).
- A result fend writes as a percentage (`80 * 15%` → `1200%`) is re-evaluated as a plain number (`12`); a leading
  `approx. ` is dropped.
- With `decimals`, the outcome is `X = <exact>, which is <rounded> at N decimals`, unless rounding changes nothing.

Rejection texts, frozen (`rejection()` in `src/lib.rs`, `posttool.REJECTIONS` in DayCare): part of the training
envelope, byte for byte.

| error | stderr |
|---|---|
| `MalformedArguments` | `malformed arguments: see the tool schema` |
| `MissingField` | `missing required field` |
| `OutOfRange` | `value out of range` |

## Timeout

Each fend evaluation is interrupted after 1 s (`9^9^9` refuses with `could not calculate ...: interrupted.`). A
`decimals` call evaluates twice, so one call is bounded by about 2 s (4 s when a percentage result is re-evaluated).

## Parity

Checked against the runner used for run 5: 2,204 unique argument strings taken from run 5's records plus 71 edge cases
(malformed JSON, wrong and extra fields, bounds, rounding, percentages, units, division by zero, huge numbers,
timeouts) give byte-identical stdout; the 1,709 recorded `calculate` results re-render byte-identical through
`posttool.Calculator`. The binary's own sha256 differs (a different crate), the bytes it returns do not.

## License

MIT, like DayCare. It links [fend-core](https://github.com/printfn/fend) (MIT, Copyright (c) 2020-2025 its authors); see
`THIRD_PARTY_NOTICES.md`. `serde`/`serde_json` are MIT OR Apache-2.0.
