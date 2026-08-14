# BoltBeam Training Dataset Idea

## Objective

Train a small policy model or adapter that can propose good BoltBeam next
actions:

```text
profile + target + search space + evidence + ledger
  -> bottleneck diagnosis
  -> candidate ranking
  -> next measurement
  -> scope draft
```

The model should not replace BoltBeam. It should rank and draft. BoltBeam still
verifies.

## Example Training Record

```json
{
  "input": {
    "model_profile": "qwen3-14b dense decoder, Q4_K attn_kv route",
    "target": "amd_gfx1100",
    "evidence": "reduce bucket 52%, attn_k route miss",
    "ledger": "split-K refuted, topology exhausted"
  },
  "target": {
    "diagnosis": "route-missed Q4_K attn_k, not FFN topology",
    "next_action": "route attn_k through generated G3 and run W==D",
    "verdict_style": "candidate until token_match and speed pass"
  }
}
```

## SFT Dataset Sources

- BoltBeam audit reports
- tinygrad result docs
- ledger entries
- candidate decisions
- roofline reports
- scope docs
- failed hypotheses with reasons

## DPO Pair Construction

Chosen answer:

- cites evidence
- respects search space
- states uncertainty
- refuses unsupported promotion
- proposes a narrow next step

Rejected answer:

- guesses from static counts only
- ignores context regression
- mixes quant/role rows
- promotes without rollback
- writes a model-specific one-off route

## RLVR Reward Functions

Good verifiable rewards:

- valid JSON schema
- candidate id exists
- candidate is in search space
- evidence model/target/workload match
- no promotion without rollback
- all cited paths exist
- verdict matches known ledger result
- generated command is syntactically valid

Risky rewards:

- "sounds plausible"
- "mentions performance"
- "longer answer"
- "uses confident tone"

## Expected Role Of The Learned Model

The learned model should improve search ordering:

```text
bad:
  learned model says promote -> ship

good:
  learned model says try candidate X first
  BoltBeam checks legality
  tinygrad measures
  BoltBeam evaluates
  ledger records
```

This mirrors the AlphaZero pattern:

```text
network = prior
search = verifier/refiner
rules = ground truth
```

For BoltBeam:

```text
model = candidate prior
BoltBeam = audit rules
tinygrad = executor/measurement
ledger = memory
```

