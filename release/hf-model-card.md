---
license: other
license_name: nvidia-nemotron-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
base_model_relation: adapter
pipeline_tag: text-generation
library_name: peft
language:
  - en
tags:
  - lora
  - peft
  - gguf
  - llama.cpp
  - reinforcement-learning
  - rloo
  - tool-use
  - nemotron
  - tinygrad
---

<!-- DRAFT. Model id: JulianAbeleda/nemotron-3-nano-4b-arkey. Not uploaded. -->

# nemotron-3-nano-4b-arkey

A LoRA adapter for [NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16), trained with
RLOO on the tool turns of one app: [GameTerm](https://gameterm.arkey.ai), a terminal app with a built-in assistant served by llama.cpp. The training
used one RTX 5090 and one inference stack (tinygrad samples and trains in the same process), and ran 102 updates in 86
minutes. The training states match the app's prompts, tool schemas and tool rejections exactly.

**It is a recipe demonstration, not a general tool-use upgrade.** On the app's held-out exam it improves the turn
*after* a tool call a lot. On public tool-use benchmarks (BFCL, When2Call) it shows no measurable change, neither a
regression nor a gain.

- Code, recipe and full records: [DayCare-Public](https://github.com/JulianAbeleda/DayCare-Public)
  ([write-up](https://github.com/JulianAbeleda/DayCare-Public/blob/main/docs/writeup.md),
  [RL training](https://github.com/JulianAbeleda/DayCare-Public/blob/main/docs/rl-training.md),
  [run-5 record](https://github.com/JulianAbeleda/DayCare-Public/blob/main/research/rloo-posttool-calculator-r5.md),
  [public benchmarks](https://github.com/JulianAbeleda/DayCare-Public/blob/main/research/public-benchmarks-r5.md))

## Files

| File | Format | Use |
|---|---|---|
| `adapter.gguf` | llama.cpp LoRA GGUF (f32 A/B, `adapter.lora.alpha` = 64) | `llama-server --lora adapter.gguf` on a GGUF of the base model |
| `adapter_model.safetensors` + `adapter_config.json` | PEFT LoRA (f32 A/B, `r` = 32, `lora_alpha` = 64) | `PeftModel.from_pretrained(base, "JulianAbeleda/nemotron-3-nano-4b-arkey")` with `transformers` |
| `adapter-raw.npz` | NumPy archive of the float32 LoRA factors (`lora.{0,1}.{A,B}`) | framework-neutral copy of the same weights, bit-identical to what was trained |

The adapter targets the MLP of the final block, layer 41: `blk.41.ffn_up.weight` / `blk.41.ffn_down.weight` in
llama.cpp (architecture `nemotron_h`), `backbone.layers.41.mixer.up_proj` / `.down_proj` in the Hugging Face
checkpoint (`model.layers.41...` in transformers 5). Rank 32, alpha 64 (scale alpha/rank = 2), 1,003,520 parameters.
The delta is `W + (alpha/r) * B @ A`. All three files hold bit-identical float32 factors.

The PEFT files were produced from `adapter.gguf` by
[`daycare/artifact/peft_export.py`](https://github.com/JulianAbeleda/DayCare-Public/blob/main/daycare/artifact/peft_export.py)
(the factors are copied, not converted: llama.cpp's `lora_a`/`lora_b` have PEFT's `lora_A`/`lora_B` shapes and the same
alpha/r scale). Checked on the real model: the base weights under the llama.cpp and HF names are bit-equal (no
permutation), and on 5 chat prompts transformers 5.17 + peft 0.21 (fp32) reproduced `llama-server --lora` (BF16 GGUF):
greedy continuations identical for 32/32 tokens on 5/5 prompts (3 of which differ from the base model's), the
adapter's shift of the top-20 next-token log-probabilities agreed within 0.013 nats (correlation >= 0.9998; the shift
itself is up to 0.74 nats), and the remaining gap to llama.cpp (<= 0.11 nats) is the same with and without the
adapter (fp32 vs BF16 kernels).

## How to use

llama.cpp (the stack the adapter was trained for and evaluated on):

```bash
# convert the base model once (llama.cpp)
python convert_hf_to_gguf.py NVIDIA-Nemotron-3-Nano-4B-BF16 --outtype bf16 --outfile nemotron-3-nano-4b-bf16.gguf

llama-server -m nemotron-3-nano-4b-bf16.gguf --lora adapter.gguf --jinja -ngl 99 -c 65536
```

transformers + PEFT:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "JulianAbeleda/nemotron-3-nano-4b-arkey")

messages = [{"role": "user", "content": "What is 17 * 23?"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=False).to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=512, do_sample=False)[0, ids.shape[1]:]))
```

The PEFT path was tested with transformers 5.17 (native Nemotron-H, no `trust_remote_code`) and peft 0.21. The base
repo's own remote code (`trust_remote_code=True`, transformers 4.x) needs `mamba-ssm` and was not tested; its module
names match the adapter's `target_modules` (`up_proj`, `down_proj`, `layers_to_transform` = [41]).

Tested with llama.cpp build b9592 on the BF16 GGUF (sha256 `2126a5e8056f4178200f3c30dcac11ac0ebae200fdb8d2b8ff86e2fa2930fcb8`),
thinking on (the chat template's default). Quantized bases were not tested. Serving parity was checked against the
training sampler on 9 held-out app states, greedy, with the same token ids: 6/9 continuations identical up to 384
tokens. The 3 divergences were near-ties (0.007 to 0.033 nats between the top two tokens).

## Intended use

- **Intended:** a small local assistant in a tool-using app, on the turn after a tool call. That means recovering when
  a calculator call is rejected, relaying a tool result as the answer, and, when the app refuses a file or terminal
  call, saying so (asking for authority at most once) rather than working around it or inventing content. Also a
  worked example for app builders who want to train their own small model on their own app's tool turns.
- **Out of scope:** a general improvement in tool selection or function calling (the public benchmarks show none);
  any safety-critical use; apps whose prompts and tools differ from GameTerm's without your own evaluation. The
  gains are measured only on GameTerm's request format.

## Evaluation

GameTerm held-out exam, adapter vs stock, the same app prompts, 4,096 tokens per turn, temperature 1.0. Paired
bootstrap 95% CIs, clustered by task:

| Scenario | n | Stock -> adapter | Diff, 95% CI |
|---|---:|---|---|
| repair: calculator rejected the call | 920 | 76.3% -> 91.8% | **+15.5 [+11.2, +20.0]** |
| relay: the result is the answer | 1,700 | 94.4% -> 95.1% | +0.8 [-0.6, +2.1] |
| blank answers, relay family | 2,620 | 97 -> 16 | **-3.1 pts [-4.1, -2.1]** |
| wrong answers, relay family | 2,620 | 217 -> 142 | -2.9 pts [-4.5, -1.3] |
| long-number relay/repair | 360 | 86.9% -> 97.8% | +10.8 [+4.2, +18.6] |
| blocked call handled correctly | 620 | 27.1% -> 39.4% | **+12.3 [+7.7, +16.9]** |
| blocked, out-of-distribution probe (78 new states) | 624 | 38.1% -> 48.7% | **+10.6 [+5.6, +15.4]** |
| Countdown puzzles (untrained) | 188 | 63.3% -> 60.6% | -2.7 [-10.1, +4.8] |
| retention, 132 unrelated items (T=0, one sample) | 132 | plain 40 -> 40; numeric net -1 | **failed as declared** |

Public benchmarks, served by `llama-server --lora` (b9592), thinking on, stock and adapter identical except the
adapter, one greedy sample per item unless noted:

| Benchmark | n | Stock | Adapter | Diff, 95% CI |
|---|---:|---:|---:|---|
| BFCL v4 non-live AST, pooled (simple / multiple / parallel / parallel_multiple) | 1,150 | 75.7 | 76.7 | +1.0 [-1.0, +3.0] |
| BFCL v4 irrelevance | 240 | 74.6 | 76.2 | +1.7 [-1.7, +5.0] |
| BFCL v4 multi_turn_base | 200 | 26.5 | 31.0 | +4.5 [-1.5, +10.5] |
| When2Call macro F1 (LLM-judge mode, substitute judge) | 300 | 68.8 | 66.3 | -2.5 [-6.4, +1.2] |
| When2Call tool hallucination, T=1 x 10 samples (lower is better) | 1,000 | 20.4 | 20.4 | +0.0 [-2.2, +2.2] |

When2Call was judged with its official judge prompt by a substitute LLM judge (blind, arms shuffled), not the paper's
GPT-4o, so its numbers are not comparable with the paper's. The hallucination rate uses no judge (a native tool call
is a call).

## Limitations

- **App-specific.** Trained and gated only on GameTerm's post-tool turns (calculator repair and relay, refused
  file/terminal calls). The public benchmarks show no change in general tool use.
- **Adopted by owner exception, not a pass.** One predeclared gate failed: numeric retention (one greedy sample per
  item) came out net -1 item. A follow-up
  [diagnosis](https://github.com/JulianAbeleda/DayCare-Public/blob/main/research/posttool-g3-numeric-diagnosis.md) found
  that result fragile: at T=1 on all 68 numeric items the adapter is level with stock (+1.5 [-1.1, +4.4] under the audited numeric rule, 8 samples per item; see the [run-5 record](https://github.com/JulianAbeleda/DayCare-Public/blob/main/research/rloo-posttool-calculator-r5.md)).
- **Known costs.** The adapter calls the calculator 5.5 points less often on numeric items [-9.4, -1.8]. One item
  shows a real reading shift. Turns are shorter (relay family 355 -> 248 tokens). Replies to blocked calls invent
  detail about the block itself more often in a blind audit (12/40 vs 7/40; not gated).
- **One seed.** A replication on a second seed was stopped early by its entropy trigger, so the result is not yet
  reproduced.
- **Not fully reproducible from public data.** The app's captured request envelope and the frozen task states are not
  published. The code, the command line, the calculator runner and the records are.
- Inherits every limitation, bias and safety consideration of the base model (see its card).

## Training details

- **Base:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, converted to a BF16 GGUF; thinking on.
- **Method:** RLOO (leave-one-out baseline), group 8, 4 states per update, 102 updates, Adam lr 2e-4, max grad norm
  1.0, KL anchor to stock 0.03 (matched aggregation), entropy 1e-3, a group length term (weight 1.0), a repetition
  penalty, temperature 1.0, up to 8 sampled turns per episode, 4,096 tokens per turn. Graded reward: wrong -1, blank
  -1.5, relay abstention -0.5. Blocked episodes are graded by rule (c): at most one authority request, then a clear
  reply that names the block; circumvention, repeated calls and invented content fail.
- **Data:** 427 frozen GameTerm post-tool states (mix repair 0.75 / relay 0.15 / blocked 0.1), each the app's real
  request with the stock model's own first tool call and the app's real reply. Calculator word problems
  and refused file/terminal requests; the held-out states are a fixed, disjoint split.
- **Stack:** [tinygrad-arkey](https://github.com/JulianAbeleda/tinygrad-arkey) `exp` at
  `bbf307f8347c166ec93b40d81c1b0bd3ce728563`, sampling and training in one process. The trainer recomputes every
  sampled token's log-probability and stops on a gap above 0.1 nats; the maximum gap in this run was 0. Before the
  run, a one-update export and reload in a fresh process was bit-exact.
- **Hardware:** one NVIDIA RTX 5090 (32 GB); 86 minutes of training. Training seed 20260930.
- **Discipline:** hypothesis, gates and stop triggers (KL, entropy, length, capped turns, reward drift) were committed
  before the first update; the run completed all 102 updates without a trigger.

## License

The base model is governed by the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)
(last modified December 15, 2025). This adapter is a Derivative Work of it and is distributed under the same
license. A copy of the license is included in this repository (`LICENSE`), as section 3(a) requires.

Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.

The DayCare training code is MIT-licensed. That license covers the code only, not these weights.

## Citation

```
@misc{nemotron-3-nano-4b-arkey,
  title  = {nemotron-3-nano-4b-arkey: app-specific RLOO for Nemotron 3 Nano 4B on its own app's tool turns},
  year   = {2026},
  url    = {https://huggingface.co/JulianAbeleda/nemotron-3-nano-4b-arkey},
  note   = {Code: https://github.com/JulianAbeleda/DayCare-Public}
}
```
