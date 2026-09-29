"""Export lossless XML LoRA matrices for llama.cpp without requantizing the base."""
from pathlib import Path
from . import codec,gguf_kv as g
from .native_lora import read_checkpoint


def export(adapter,output,record):
    manifest,tensors=read_checkpoint(adapter,record)
    if not manifest.architecture:raise ValueError('adapter architecture required')
    pairs=[('general.type',8,'adapter'),('general.architecture',8,manifest.architecture),
           ('adapter.type',8,'lora'),('adapter.lora.alpha',6,float(manifest.alpha))]
    payloads=[];offset=0
    for item in manifest.target_map:
        for suffix,key in [('lora_a','A'),('lora_b','B')]:
            arr=tensors[f"lora.{item['index']}.{key}"]
            quant=codec.resolve_quant(arr,'lossless');spec=codec.QUANT[quant]
            if spec.ggml_type is None:raise ValueError('adapter dtype cannot be exported losslessly to GGUF')
            raw,_=codec.quantize(arr,quant)
            payloads.append((item['tensor']+'.'+suffix,arr.shape,spec.ggml_type,offset,raw))
            offset+=(len(raw)+g.ALIGNMENT-1)//g.ALIGNMENT*g.ALIGNMENT
    if len(payloads)!=len(tensors):raise ValueError('target map does not cover all adapter tensors')
    with Path(output).open('xb') as f:
        f.write(g.MAGIC);g.w_i32(f,g.VERSION);g.w_u64(f,len(payloads));g.w_u64(f,len(pairs))
        for key,typ,value in pairs:g.w_kv(f,key,typ,value)
        for name,shape,typ,off,_ in payloads:
            g.w_str(f,name);g.w_u32(f,len(shape))
            for dim in reversed(shape):g.w_u64(f,int(dim))
            g.w_i32(f,typ);g.w_u64(f,off)
        f.write(b'\0'*((-f.tell())%g.ALIGNMENT));start=f.tell()
        for _,_,_,off,raw in payloads:
            assert f.tell()==start+off
            f.write(raw);f.write(b'\0'*((-len(raw))%g.ALIGNMENT))
    # Verify serialized matrices against the authoritative lossless XML.
    _,infos,start=g.read_gguf(str(output))
    with Path(output).open('rb') as f:
        for info,(name,shape,typ,off,raw) in zip(infos,payloads,strict=True):
            assert info==(name,tuple(reversed(shape)),typ,off),info
            f.seek(start+off)
            if f.read(len(raw))!=raw:raise ValueError('LoRA GGUF payload changed')
