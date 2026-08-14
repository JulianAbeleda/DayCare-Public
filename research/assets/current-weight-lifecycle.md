# Current Weight Lifecycle

Current training artifacts usually become maintainable through convention rather
than through one canonical file contract.

Mermaid source:
[current-weight-lifecycle.mmd](current-weight-lifecycle.mmd)

SVG render:
[current-weight-lifecycle.svg](current-weight-lifecycle.svg)

![Current weight lifecycle](current-weight-lifecycle.svg)

```mermaid
flowchart LR
  data_files["Data files"]
  raw_data["Raw data"]
  dataset_scripts["Dataset scripts"]
  processed_shards["Processed dataset shards"]

  json_files["JSON config files"]
  config_json["config.json"]
  generation_json["generation_config.json"]
  training_args_json["training_args.json"]
  safetensors_index_json["model.safetensors.index.json"]

  tokenizer_files["Tokenizer files"]
  tokenizer_json["tokenizer.json"]
  tokenizer_model["tokenizer.model"]
  special_tokens_json["special_tokens_map.json"]

  weight_files["Weight files"]
  checkpoints["Checkpoint shards"]
  safetensors["model-00001-of-000NN.safetensors"]

  doc_log_files["Docs and logs"]
  model_card["README / model card"]
  eval_logs["Eval logs"]

  code_files["Code and scripts"]
  train["Training code"]
  exporter["Conversion / exporter scripts"]

  runtime_files["Runtime output files"]
  gguf["GGUF"]
  hf["HF repo files"]
  onnx["ONNX"]
  runtime["Runtime-specific files"]

  data_files --> raw_data --> dataset_scripts --> processed_shards --> train
  data_files --> dataset_scripts
  data_files --> processed_shards

  json_files --> config_json
  json_files --> generation_json
  json_files --> training_args_json
  json_files --> safetensors_index_json

  tokenizer_files --> tokenizer_json
  tokenizer_files --> tokenizer_model
  tokenizer_files --> special_tokens_json

  weight_files --> checkpoints
  weight_files --> safetensors

  doc_log_files --> model_card
  doc_log_files --> eval_logs

  code_files --> train
  code_files --> exporter

  runtime_files --> gguf
  runtime_files --> hf
  runtime_files --> onnx
  runtime_files --> runtime

  config_json --> train
  generation_json --> train
  training_args_json --> train
  tokenizer_json --> train
  tokenizer_model --> train
  special_tokens_json --> train

  train --> checkpoints
  checkpoints --> safetensors
  safetensors --> safetensors_index_json

  config_json --> exporter
  generation_json --> exporter
  tokenizer_json --> exporter
  tokenizer_model --> exporter
  special_tokens_json --> exporter
  safetensors --> exporter
  safetensors_index_json --> exporter
  model_card -.-> exporter
  eval_logs -.-> exporter

  exporter --> gguf
  exporter --> hf
  exporter --> onnx
  exporter --> runtime

  human["Human/tool reconstructs lifecycle"] -.-> config_json
  human -.-> tokenizer_json
  human -.-> safetensors
  human -.-> model_card
  human -.-> exporter
```

```text
raw data
  -> dataset scripts
  -> processed dataset shards
  -> tokenizer files
  -> training code
  -> model config JSON
  -> checkpoint shards
  -> safetensors or framework weights
  -> exporter metadata
  -> GGUF / HF / ONNX / runtime-specific files
```

The useful pieces exist, but the relationships are spread across several files:

```text
config.json
tokenizer.json
tokenizer.model
special_tokens_map.json
generation_config.json
model.safetensors.index.json
model-00001-of-000NN.safetensors
training_args.json
README.md / model card
eval logs
conversion scripts
```

The weight lifecycle is therefore reconstructed by tools and humans:

```text
which data produced the weights?
which tokenizer was used?
which architecture matches the tensor names?
which checkpoint is canonical?
which evals justify the export?
which conversion path produced the GGUF?
```

That is workable, but it is not a flat artifact. The model is a directory of
related state, and the meaning of the final weights depends on conventions around
JSON, safetensors, scripts, and README/model-card text.
