# Data Curation And Evaluation

Data is the controlling variable for every training method in DayCare. Bad data turns every method into a behavior
copying machine. Good data lets smaller, cheaper methods work before expensive training is needed.

## Why Data Curation Is Its Own Topic

Pretraining, continued pretraining, SFT, DPO, and RLVR all fail differently, but they share one rule: the model learns
the distribution it sees. Data quality controls:

- what knowledge becomes durable;
- which formats become default behavior;
- which mistakes become reinforced;
- whether evaluations measure generalization or memorization;
- whether preference/RL rewards encode the real objective.

## Core References

- The Pile introduced a diverse 825 GiB English corpus with 22 components and documented concerns in the data:
  <https://arxiv.org/abs/2101.00027>
- Dolma released a 3T-token open corpus and a data curation toolkit for pretraining research:
  <https://arxiv.org/abs/2402.00159>
- DataComp-LM gives a controlled benchmark for filtering, deduplication, data mixing, and training-set design:
  <https://arxiv.org/abs/2406.11794>
- Benchmark contamination survey:
  <https://arxiv.org/html/2406.04244v1>
- Soft contamination shows that semantic duplicates can bypass exact-match filters:
  <https://arxiv.org/html/2602.12413v1>

## Curation Checklist

For a DayCare/BoltBeam dataset, each row should have:

- source artifact path or commit;
- task type: classify, scope, verdict, plan, refusal, rewrite, command selection;
- model/target context if the example depends on hardware or route family;
- evidence authority: local microgate, W==D, route-bound, correctness, roofline, ledger;
- verdict label;
- known invalid alternatives when available;
- split label: train, validation, or final holdout.

Do not mix final promotion gates into training data. Keep final holdouts locked.

## Evaluation Checklist

A useful evaluation should ask whether the model can:

- preserve schema validity;
- cite existing artifacts only;
- identify missing evidence;
- refuse unsupported promotion;
- keep correctness, route-bound, and speed evidence separate;
- select the next candidate inside the known search space;
- avoid changing default-on state without required gates;
- produce a measurement plan that can actually run.

## Contamination Risks

BoltBeam has a special contamination risk: the ledger is both the best training data and the strongest evaluation
authority. If the final exam is copied into training, the model can appear to reason while only memorizing verdicts.

Mitigations:

- split by candidate family and date, not random rows only;
- hold out whole route families;
- hold out recent commits;
- hold out negative examples;
- keep a small private final set of synthetic candidates with known verifiable outcomes;
- report train/validation/test provenance in every model card.

## Practical Position

Before any training run, build a data card. It should say:

- what data went in;
- what was excluded;
- what leak checks ran;
- what the model is supposed to learn;
- what the model is not allowed to claim;
- which evals decide whether the run was useful.
