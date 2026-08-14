# Running the substrate on another machine

The Substrate is inference-only tissue (`daycare/substrate/base.py`) — it holds
no state and is forgotten between calls. Nothing about that requires it to run
in the same process, or on the same computer, as the model that borrows it.

That makes one useful split possible: **the laptop keeps the interface and the
artifacts, the GPU host keeps the weights.**

| | laptop | GPU host |
|---|---|---|
| the model — `brain`, drives, `state/*.xml` | ● | |
| `artifact/` — manifest, package, save | ● | |
| the `.xpkg` and the report card | ● | |
| substrate inference | | ● |
| `nursery/` — teaching, LoRA, packaging | | ● |

The model is the thing with an identity and a history, so it belongs where you
work. The weights are fungible and heavy, so they belong where the VRAM is. The
line between them is exactly the Substrate protocol, which is why this costs one
class rather than an architecture.

Note the asymmetry in the last row: `nursery/lesson.py` imports the tinygrad
trainer at module scope, so **teaching cannot run on a machine without CUDA at
all** — not even as a client. Its outputs (GGUFs, `.xpkg` directories) are
runtime data excluded by `.gitignore`, so they come back over `rsync`, not over
a commit. That half of the split is not yet automated.

## Serving

`llama_cpp.py` spawns its own server and binds loopback, which is right for a
machine you are sitting at and useless across a network. Start a long-lived one
on the GPU host instead:

```bash
llama-server \
  --model ~/models/Ada-Qwen3-0.6B-Q8_0.gguf \
  --alias daycare-substrate \
  --host <tailnet-ip> --port 8081 \
  --ctx-size 4096 --gpu-layers all --parallel 1 \
  --api-key-file ~/.config/daycare/substrate.key
```

Two deliberate choices:

- **Bind the tailnet address, not `0.0.0.0`.** The interface *is* the access
  control — the endpoint is unreachable from anywhere but the tailnet, and the
  API key is the second lock rather than the only one. Never put this on a
  public interface; llama-server has no rate limiting and loads whatever you
  point it at.
- **Use `--api-key-file`, not `--api-key`.** Arguments are visible in `ps` to
  every user on the box; a `0600` file is not.

Port 8081 because 8080 may already be serving something else — on this tailnet
it is Arkey's `arkey-local` instance, and clobbering it would be rude.

## Connecting

```bash
export DAYCARE_REMOTE_URL=http://<tailnet-ip>:8081
python -m daycare.nursery.evaluate ~/models/Qwen3-0.6B-Q8_0.gguf Ada
```

The key is read from `~/.config/daycare/substrate.key` if it exists, so it
normally needs no environment variable at all.

| variable | |
|---|---|
| `DAYCARE_REMOTE_URL` | server root, e.g. `http://100.x.y.z:8081` (required) |
| `DAYCARE_REMOTE_KEY` | bearer token, overriding the key file |
| `DAYCARE_REMOTE_KEY_FILE` | where to read it instead (default above) |
| `DAYCARE_REMOTE_TIMEOUT` | per-request seconds (default 600) |

## What comes over the wire

One prompt out, one completion back, per turn — a few kilobytes of JSON. The
belief block, the drives, the enforcement ladder, and every XML write stay on
the laptop, because the Substrate never sees them; it receives a finished prompt
and a temperature. Over a LAN tailnet the round trip is not the thing you notice.

The identity gate survives the trip. `gated_identity` reads the served model's
chat template through `/props`, so an identity-gated GGUF behaves the same way
remotely as it does locally, and the brain drops its belief block accordingly.

## What this is not

It is not the orchestrator. The Substrate protocol is `(prompt, controls) ->
text`; the model that drives an application shell needs message history, tool
calls, and structured output, and forcing that through `generate()` would mean
scraping strings. That is a second seam and a second (larger) model, served the
same way from the same host — but it is not this one.
