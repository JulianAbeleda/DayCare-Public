import copy
import json

import numpy as np
import pytest

from daycare.nursery import trainer_env
from daycare.artifact.native_lora import (
    checkpoint_path, digest, read_checkpoint, validate_metadata, write_checkpoint,
)


def example():
    tensors = {'lora.0.A': np.array([[0.12345679, -0.9876543, 1e-7]], dtype=np.float32),
               'lora.0.B': np.array([[0.31415927], [-0.27182818]], dtype=np.float32)}
    record = dict(schema='daycare.native_tool_sft.v1', model_sha256='base-hash', rank=1,
                  alpha=2.5, last_k=1, epochs=1, losses=[{'loss': 1.0}],
                  model_profile={'architecture': 'fixture'}, dataset_sha256='dataset-hash',
                  target_map=[dict(index=0, layer=0, target='ffn_up',
                                   tensor='blk.0.ffn_up.weight', shape=[2, 3])])
    return tensors, record


@pytest.mark.skipif(not trainer_env.trainer_available(),
                    reason='trainer tinygrad not installed')
def test_lora_gguf_preserves_matrices_shapes_and_alpha(tmp_path):
    from daycare.artifact.lora_gguf import export
    from daycare.artifact.gguf_kv import read_gguf
    tensors,record=example();source=tmp_path/'adapter.xml';out=tmp_path/'adapter.gguf'
    write_checkpoint(source,tensors,record);record['adapter_sha256']=digest(source)
    export(source,out,record)
    kv,infos,start=read_gguf(str(out));metadata={k:v for k,_,v in kv}
    assert metadata['adapter.lora.alpha']==2.5 and metadata['general.type']=='adapter'
    assert [i[1] for i in infos]==[(3,1),(1,2)]
    with out.open('rb') as stream:
        for info,key in zip(infos,['lora.0.A','lora.0.B']):
            stream.seek(start+info[3]);assert stream.read(tensors[key].nbytes)==tensors[key].tobytes()
    with pytest.raises(ValueError,match='base model'):
        export(source,tmp_path/'bad.gguf',dict(record,model_sha256='other'))


def test_xml_is_lossless_with_fractional_alpha_and_rejects_mismatched_identity(tmp_path):
    tensors, record = example()
    path = tmp_path / 'adapter.xml'
    write_checkpoint(path, tensors, record)
    record.update(adapter_file='adapter.xml', adapter_sha256=digest(path))
    manifest, restored = read_checkpoint(checkpoint_path(tmp_path, record), record)
    assert manifest.alpha == 2.5 and manifest.eval_gate == 'pending'
    assert 'quant="f32"' in path.read_text()
    for key, tensor in tensors.items():
        assert restored[key].tobytes() == tensor.tobytes()
    for key, value in [('alpha', 3), ('rank', 2), ('model_sha256', 'wrong')]:
        with pytest.raises(ValueError, match='differs'):
            validate_metadata(manifest, dict(record, **{key: value}))
    wrong = copy.deepcopy(record)
    wrong['target_map'][0]['tensor'] = 'blk.7.ffn_up.weight'
    with pytest.raises(ValueError, match='target map'):
        validate_metadata(manifest, wrong)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash changed'):
        read_checkpoint(path, record)


def test_legacy_checkpoint_maps_to_same_merge_delta(tmp_path):
    tensors, record = example()
    np.savez(tmp_path / 'adapter.npz', **{k.replace('lora.', '').replace('.', '_'): v for k, v in tensors.items()})
    write_checkpoint(tmp_path / 'adapter.xml', tensors, record)
    (tmp_path / 'run.json').write_text(json.dumps(dict(record, adapter_sha256=digest(tmp_path / 'adapter.npz'))))
    _, legacy = read_checkpoint(checkpoint_path(tmp_path, {}))
    _, xml = read_checkpoint(tmp_path / 'adapter.xml')
    np.testing.assert_array_equal(legacy['lora.0.B'] @ legacy['lora.0.A'],
                                  xml['lora.0.B'] @ xml['lora.0.A'])
    with pytest.raises(ValueError, match='filename'):
        checkpoint_path(tmp_path, {'adapter_file': '../adapter.xml'})


def test_mixed_tensor_dtypes_derive_storage_without_narrowing(tmp_path):
    from daycare.artifact import codec, index
    tensors, record = example()
    tensors['lora.0.A'] = tensors['lora.0.A'].astype(np.float16)
    path = tmp_path / 'mixed.xml'
    write_checkpoint(path, tensors, record)
    manifest, restored = read_checkpoint(path)
    _, specs = index.from_element(manifest.tensors_el)
    assert {s.name: s.quant for s in specs} == {'lora.0.A': 'f16', 'lora.0.B': 'f32'}
    for name in tensors:
        assert codec.lossless_equal(tensors[name], restored[name])
    tensors['lora.0.A'] = tensors['lora.0.A'].astype(np.float64)
    with pytest.raises(ValueError, match='no lossless artifact codec'):
        write_checkpoint(tmp_path / 'unsupported.xml', tensors, record)
    assert not (tmp_path / 'unsupported.xml').exists()
