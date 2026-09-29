"""Native-tool LoRA checkpoint boundary: dtype-derived lossless XML, legacy NPZ reads."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from . import codec
from .load import load_adapter
from .save import write_adapter


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checkpoint_path(directory, record):
    name = record.get('adapter_file', 'adapter.npz')
    if Path(name).name != name or name not in ('adapter.xml', 'adapter.npz'):
        raise ValueError('invalid adapter filename in training record')
    return Path(directory) / name


def validate_metadata(manifest, record):
    """Do not apply shape-compatible matrices to a different base, target, or scale."""
    for actual, expected, label in (
        (manifest.r, record['rank'], 'rank'),
        (manifest.alpha, record['alpha'], 'alpha'),
        (manifest.source_sha256, record['model_sha256'], 'base model'),
        (json.dumps(manifest.target_map, sort_keys=True),
         json.dumps(record['target_map'], sort_keys=True), 'target map'),
    ):
        if actual != expected:
            raise ValueError(f'adapter {label} differs from training configuration')


def read_checkpoint(path, record=None):
    path = Path(path)
    if record is not None and digest(path) != record['adapter_sha256']:
        raise ValueError('adapter hash changed')
    if path.suffix == '.xml':
        manifest, tensors = load_adapter(str(path))
        if record is not None:
            validate_metadata(manifest, record)
        return manifest, tensors
    if path.suffix != '.npz':
        raise ValueError('adapter must be XML or legacy NPZ')
    if record is None:
        sidecar = next((path.parent / name for name in ('run.json', 'training-progress.json')
                        if (path.parent / name).is_file()), None)
        if sidecar is None:
            raise ValueError('legacy adapter requires a provenance sidecar')
        record = json.loads(sidecar.read_text())
        if digest(path) != record.get('adapter_sha256'):
            raise ValueError('legacy adapter hash changed')
    required = ('rank', 'alpha', 'model_sha256', 'target_map')
    if any(not record.get(key) for key in required):
        raise ValueError('legacy adapter provenance is incomplete')
    with np.load(path, allow_pickle=False) as saved:
        tensors = {}
        for key in saved.files:
            index, sep, name = key.partition('_')
            if not sep or not index.isdigit() or name not in ('A', 'B'):
                raise ValueError('invalid legacy adapter tensor name')
            tensors[f'lora.{int(index)}.{name}'] = saved[key].copy()
    return SimpleNamespace(r=record["rank"], alpha=record["alpha"],
                           source_sha256=record["model_sha256"], target_map=record["target_map"]), tensors


def write_checkpoint(path, tensors, record):
    """Request lossless storage; codec policy owns concrete tensor dtypes."""
    if any(not np.isfinite(value).all() for value in tensors.values()):
        raise ValueError('native LoRA checkpoint requires finite tensors')
    losses = record['losses']
    meta = dict(subject=record.get('training_objective', 'native-tool-sft'), base=record['model_sha256'],
                r=record['rank'], alpha=record['alpha'], last_k=record['last_k'],
                targets=list(dict.fromkeys(t['target'] for t in record['target_map'])),
                epochs=record['epochs'], loss_start=losses[0]['loss'], loss_end=losses[-1]['loss'],
                eval_metric='external-task-gate', eval_before='not-evaluated',
                eval_after='not-evaluated', eval_gate='pending', method='lora',
                source_sha256=record['model_sha256'],
                architecture=record['model_profile']['architecture'], target_map=record['target_map'])
    write_adapter(tensors, meta, str(path), quant='lossless')
    tree = ET.parse(path)
    provenance = ET.SubElement(tree.getroot(), 'provenance', schema=record['schema'])
    for key in ('daycare_revision', 'runner_sha256', 'dataset_sha256', 'envelope_sha256',
                'initial_adapter_sha256', 'seed', 'lr', 'examples',
                'training_objective', 'reward_verifier_sha256'):
        if record.get(key) is not None:
            ET.SubElement(provenance, 'field', name=key, value=str(record[key]))
    ET.indent(tree)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    manifest, restored = read_checkpoint(path)
    validate_metadata(manifest, record)
    if set(restored) != set(tensors) or any(not codec.lossless_equal(v, restored[k]) for k, v in tensors.items()):
        raise ValueError('XML adapter did not round-trip exactly')


def validate_export_base(directory, record):
    """Require the upstream conversion's identity record before a native HF merge."""
    directory = Path(directory)
    lineage = json.loads((directory / 'daycare-base-lineage.json').read_text())
    if (lineage.get('gguf_sha256') != record['model_sha256'] or
            lineage.get('safetensors_sha256') != digest(directory / 'model.safetensors')):
        raise ValueError('base export lineage does not match training base')
