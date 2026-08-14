# Generic image-style training

DayCare treats image-style learning as a trainer-neutral project type. A project
describes the dataset, captions, holdout material, base model, trigger token,
training configuration, samples, and evaluations. It does not encode a specific
art style, franchise, character, image generator, or UI.

```text
source images
  -> curated train and holdout splits
  -> paired captions
  -> project validation
  -> external image LoRA trainer
  -> fixed-seed samples
  -> evaluations and run ledger
  -> inference UI such as ComfyUI
```

The external trainer boundary is intentional. DayCare's existing tinygrad LoRA
implementation targets language-model transformer layers; diffusion trainers
also need resolution buckets, VAE latent caching, noise schedules, U-Net and
text-encoder adapters, and safetensors export. DayCare owns the reproducible
project contract while a compatible backend performs numerical training.

## Create a project

Keep runtime projects and model artifacts outside the Git checkout:

```bash
python -m daycare.image init ~/daycare-image-projects/example-style \
  --name example-style \
  --trigger dcstyle_example \
  --base-model OnomaAIResearch/Illustrious-XL-v1.1
```

This creates:

```text
example-style/
  project.json
  training.toml
  data/train/       image + same-stem .txt caption pairs
  data/holdout/     image + same-stem .txt caption pairs
  samples/
  evaluations/
  runs/
```

Validate before training:

```bash
python -m daycare.image validate ~/daycare-image-projects/example-style
```

The default gate requires at least 20 captioned training images and one
captioned holdout image. Projects may raise the minimum in `project.json`.

## Captioning principle

Caption the content that should remain controllable: subject, composition,
pose, clothing, palette, and background. Do not redundantly caption the shared
visual traits that the project trigger is meant to learn. Dataset provenance
and permission remain project responsibilities.
