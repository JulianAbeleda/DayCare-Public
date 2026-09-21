# DayCare LoRA Training Standard

This is the default protocol for new DayCare supervised LoRA work. A task may
add stricter gates, but it must state and justify any departure from these
rules. The native tool implementation is `daycare.nursery.native_tool_sft`.

## Training contract

1. Define the task, score, adoption threshold, retention probes, and deciding
   holdout before optimization. Freeze and hash the deciding data.
2. Separate development data from the deciding holdout. Mine failure *types*
   into new examples without copying held-out requests or labels from the model
   being trained.
3. Use the model's native tokenizer and chat template. Supervise only the next
   assistant completion; prompt, system, tool-schema, prior assistant, and tool
   result tokens are context. Tool-loop tasks must include successful
   post-result turns rather than only their initial calls.
4. Keep base weights frozen and record every adapted tensor, rank, alpha,
   learning rate, epochs, seed, initial-adapter identity, and input hash.
5. Validate structured targets with their executable schema or application
   normalizer before training. Cover accepted fallback actions under the input
   and retrieval conditions in which they will be needed.
6. Mix retention examples into training when the task can change routing or
   response mode. Tool training requires at least 25% ordinary no-tool answer
   examples; answers conditioned on tool history do not count toward this
   quota. Spread each group's quota throughout the epoch rather than exhausting
   smaller retention groups first.
7. Compare before and after on identical inference settings. Report paired
   gains and losses, the standard deviation and standard error of item-level
   differences, and a paired interval. Report per-task results rather than
   hiding them in one average.
8. Reject a candidate that misses a predeclared primary or retention gate.
   Statistical uncertainty is context for the size of a change, not permission
   to rename a measured loss as an improvement.
9. Keep training outputs and weights outside Git. Commit the reusable runner,
   frozen protocol description, and an honest result record.

## Research basis

LoRA freezes pretrained weights and learns low-rank updates, which is the update
implemented by `daycare.nursery.lora`.

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models":
  <https://arxiv.org/abs/2106.09685>

Completion-only instruction tuning follows the response-only supervision used
in instruction-following work: the input defines the condition and loss is
applied to the desired response.

- Ouyang et al., "Training language models to follow instructions with human
  feedback": <https://arxiv.org/abs/2203.02155>

The separation of tuning data from held-out evaluation and the need to report
uncertainty follow established empirical ML practice. DayCare uses paired
comparisons because both model versions answer the same items.

- Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in
  Natural Language Processing": <https://aclanthology.org/P18-1128/>

For tool tasks, ToolLLM supports separating retrieval from tool-use training,
and ToolACE supports diverse, verified function-call examples.

- Qin et al., "ToolLLM": <https://arxiv.org/abs/2307.16789>
- Liu et al., "ToolACE": <https://arxiv.org/abs/2409.00920>

DayCare's local evidence is the first retrieved-tool continuation run. It
rejected a 246/256 candidate against a 251/256 starting adapter and showed that
single canonical labels can erase useful fallback behavior. The full result is
in `research/nemotron-retrieval-training/RESULTS.md`.

For calculator tasks, the [verified-training research note](calculator-verified-training.md)
documents Toolformer, ToRA, QLoRA, their applicability limits, and a frozen
small-test protocol. Require actual execution and equality with an independent
answer key, and review whether expressions include all necessary calculations.
The 25% retention quota and scheduling policy are local choices, not thresholds
established by these papers.
