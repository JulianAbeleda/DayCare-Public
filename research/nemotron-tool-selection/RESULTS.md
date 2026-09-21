# Nemotron tool selection before/after the GameTerm harness

Date: 2026-09-20  
Base: NVIDIA Nemotron 3 Nano 4B Q4_K_M  
Candidate: GameTerm rank-4 final-MLP adapter, merged and quantized Q4_K_M

## Result

A frozen 2×2 evaluation compared base versus candidate under a short native
tool context and GameTerm's full 35-tool harness. Each arm received the same 320
requests: 40 each for calculation, terminal, files, apps, media, windows, web,
and no-tool answers. Temperature was zero and no proposed tool was executed.

| Model | Short context | Full GameTerm harness |
| --- | ---: | ---: |
| Base | 263/320 (82.2%) | 144/320 (45.0%) |
| Candidate | 304/320 (95.0%) | 303/320 (94.7%) |

Training improved short-context selection by **12.81 percentage points**
(paired item SD 37.03, SE 2.07, stratified-bootstrap 95% CI 9.38 to 16.25;
45 improvements and four losses, exact paired p = `8.23e-10`). Under the full
harness it improved selection by **49.69 points** (SD 50.08, SE 2.80, CI 45.31
to 54.06; 159 improvements and no losses, p = `2.74e-48`).

The harness reduced the base by **37.19 points** (SD 53.34, SE 2.98, CI -42.19
to -32.19), but changed the candidate by only **-0.31 points** (SD 25.66, SE
1.43, CI -2.81 to 2.19; exact paired p = 1.0). The paired difference-in-
differences was **+36.88 points** (SD 55.00, SE 3.07, CI 31.56 to 42.19).
This supports the experiment's central claim: the adapter removed the tool-
selection suppression caused by GameTerm's long harness on this suite.

## Per-category scores

| Group | Base short | Candidate short | Base harness | Candidate harness |
| --- | ---: | ---: | ---: | ---: |
| Calculate | 38/40 | 40/40 | 22/40 | 34/40 |
| Terminal | 29/40 | 32/40 | 19/40 | 40/40 |
| Files | 39/40 | 39/40 | 23/40 | 39/40 |
| Apps | 33/40 | 34/40 | 1/40 | 30/40 |
| Media | 31/40 | 40/40 | 27/40 | 40/40 |
| Windows | 14/40 | 40/40 | 4/40 | 40/40 |
| Web | 40/40 | 40/40 | 8/40 | 40/40 |
| No tool | 39/40 | 39/40 | 40/40 | 40/40 |

The candidate retained no-tool abstention while gaining every action category
under the full harness. Per-category full-harness improvements ranged from 12
of 40 calculation items to 36 of 40 window items; no category lost accuracy.

## Fend and argument validity

GameTerm's `calculate` tool uses the compiled-in `fend-core` engine. The model
evaluation itself executed nothing. Afterwards, the 136 saved `calculate`
proposals were replayed through GameTerm's real `calculate::normalize` and
`calculate::evaluate` path with fallbacks disabled. Every call ran in fend and
produced the correct value for its frozen request.

| Math outcome | Base short | Candidate short | Base harness | Candidate harness |
| --- | ---: | ---: | ---: | ---: |
| Correct through fend | 38/40 | 40/40 | 22/40 | 34/40 |
| Correct by fend or direct answer | 38/40 | 40/40 | 29/40 | 34/40 |

Under the full harness, correct fend-assisted solutions rose by **30 points**
(paired SD 46.41, SE 7.34, bootstrap 95% CI 17.5 to 45.0; 12 gains, no losses,
exact paired p = `0.00049`). Counting correct direct answers too, user-visible
solutions rose from **72.5% to 85.0%**, a **12.5-point** change (SD 33.49, SE
5.30, bootstrap CI 2.5 to 22.5). That comparison has five gains and no losses;
its conservative exact paired p is `0.0625`, so a larger end-to-end math set is
needed for a stronger total-solution claim.

Across all tools, 904 first calls were proposed and every argument object passed
its published JSON schema. Provider tools have no local typed normalizer.

## Frozen data and claim boundary

The suite SHA-256 is
`ceba3e599cdfb088d540aa720975bf2346e575046cda95d74eb29b8de15de30b`.
It was frozen before either expanded arm ran and has no exact request overlap
with the 304 training examples or original 24-question quiz. Suite, scripts,
raw responses, and bootstrap output remain outside Git under
`<server>/storage/daycare-runs/nemotron-tool-selection-002/`.

This is enough to claim a tool-selection improvement for this adapter, a
measured interaction with the GameTerm harness, and improved fend-assisted
accuracy on these 40 arithmetic requests. The replay exercises GameTerm's real
Rust normalization and fend engine, but not the native app's transcript/UI path.
One training seed does not establish that the recipe reproduces across runs.
The deciding MacBook Air smoke and live app effect checks remain unmeasured.
