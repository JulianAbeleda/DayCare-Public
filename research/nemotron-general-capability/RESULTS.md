# Nemotron tool adapter: general-capability before/after

Date: 2026-09-20  
Base: NVIDIA Nemotron 3 Nano 4B Q4_K_M  
Candidate: GameTerm rank-4 final-MLP adapter, merged and quantized Q4_K_M

## Result

Most tested capability remains, but unchanged general capability is not
supported. On a frozen 140-item generated-answer multiple-choice suite, the
candidate scored **85/140 (60.7%)** versus the base's **98/140 (70.0%)** under a
minimal prompt. The paired change was **-9.29 percentage points**, with paired
item SD **31.50 points**, SE **2.66 points**, and a stratified paired-bootstrap
95% interval of **[-14.29, -4.29]**. There was one improved item and 14 worsened
items (exact paired McNemar p = **0.00098**).

Under GameTerm's full 35-tool envelope, the candidate scored **75/140 (53.6%)**
versus **83/140 (59.3%)**. The paired change was **-5.71 points**, paired item SD
**31.21**, SE **2.64**, and bootstrap interval **[-10.71, -0.71]**. There were
three improvements and 11 losses; the conservative exact paired test was
borderline (p = **0.057**). No model proposed a tool on these questions.

The result establishes a generated multiple-choice accuracy loss on this frozen
sample, especially without the GameTerm envelope. It does not establish a
universal loss of “general intelligence.”

## Frozen suite

Before either arm ran, seed `20260920` selected 180 public test/validation rows:

| Task | Items | Source |
| --- | ---: | --- |
| MMLU | 60 | three test rows from each of 20 subjects (`cais/mmlu`) |
| ARC-Challenge | 40 | test (`allenai/ai2_arc`) |
| HellaSwag | 40 | validation (`Rowan/hellaswag`) |
| GSM8K | 40 | test (`openai/gsm8k`) |

The canonical suite SHA-256 is
`7e927f93bb5cb22327e5bcb33f3463419dd4f113df8c2a719d2809020264f7b1`.
It has no exact request overlap with the 304 training rows. The suite, selection
code, raw responses, and statistics remain outside Git under
`<server>/storage/daycare-runs/nemotron-general-capability-001/`.

## Per-task scores

| Setting and task | Base | Candidate | Change | Paired SD | 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minimal MMLU | 43/60 | 37/60 | -10.0 pp | 35.42 pp | [-20.0, -1.67] |
| Minimal ARC-Challenge | 34/40 | 34/40 | 0.0 pp | 0.0 pp | [0.0, 0.0] |
| Minimal HellaSwag | 21/40 | 14/40 | -17.5 pp | 38.48 pp | [-30.0, -7.5] |
| Minimal GSM8K | 4/40 | 4/40 | 0.0 pp | 39.22 pp | [-12.5, 12.5] |
| GameTerm MMLU | 35/60 | 35/60 | 0.0 pp | 31.89 pp | [-8.33, 8.33] |
| GameTerm ARC-Challenge | 33/40 | 30/40 | -7.5 pp | 26.67 pp | [-17.5, 0.0] |
| GameTerm HellaSwag | 15/40 | 10/40 | -12.5 pp | 33.49 pp | [-22.5, -2.5] |

GSM8K used a terse zero-shot final-number prompt and both arms were near the
floor. Its net tie is inconclusive, so it is excluded from the primary
multiple-choice aggregate. GSM8K was not run inside GameTerm because using the
calculator there would be correct tool behavior rather than an intelligence
failure.

## What changed

The candidate acquired a pronounced first-option bias. The correct-answer mix
contained 30 A answers among 140 items. The base selected A 43 times under the
minimal prompt and 55 times under GameTerm; the candidate selected it 65 and 78
times. Manual review of every discordant response found direct answer changes,
not parser or tool-call artifacts.

This suite uses zero-shot generated letters rather than each benchmark's
official likelihood-scoring protocol, and it is a subset of each benchmark.
The absolute numbers must not be compared with leaderboard scores. Its value is
the paired base/candidate comparison with identical frozen prompts. Temperature
zero makes each response reproducible; the reported SD measures variation in
paired outcomes across questions, not repeated-sampling randomness.
