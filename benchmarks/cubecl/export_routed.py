#!/usr/bin/env python3
"""Export all experts and balanced/skewed top-8 routes into an ignored local fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'crates/llama-diffusion-sys/vendor/llama.cpp/gguf-py'))
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize
args.output.mkdir(parents=True, exist_ok=False)
tensor = next(t for t in GGUFReader(args.model).tensors if t.name == 'blk.0.ffn_gate_up_exps.weight')
assert tensor.tensor_type == GGMLQuantizationType.Q4_K
assert list(tensor.shape) == [2816, 1408, 128]
k, n, experts, top_k = 2816, 1408, 128, 8
packed = tensor.data.view(np.uint8).reshape(experts, -1)
packed.tofile(args.output/'experts.q4k')
packed_hash = hashlib.sha256(packed.tobytes()).hexdigest()
cases, inputs, routes, expected = [], [], [], []
for m in [128, 512]:
    x = np.random.default_rng(42+m).standard_normal((m,k)).astype('<f4')
    for distribution in ['balanced', 'skewed']:
        ids = np.arange(m*top_k).reshape(m,top_k) % (experts if distribution == 'balanced' else top_k)
        ids = ids.astype('<u4')
        name = f'routed_{distribution}_{m}'
        x.tofile(args.output/(name+'.input.f32'))
        ids.tofile(args.output/(name+'.ids.u32'))
        counts = np.bincount(ids.reshape(-1), minlength=experts)
        cases.append({'name':name,'m':m,'k':k,'n':n,'experts':experts,'top_k':top_k,
                      'packed':'experts.q4k','packed_sha256':packed_hash,
                      'input':name+'.input.f32','ids':name+'.ids.u32','expected':name+'.expected.f32',
                      'expert_counts':counts.tolist(),'distribution':distribution})
        inputs.append(x.astype(np.float64));routes.append(ids);expected.append(np.empty((m,top_k,n),dtype='<f4'))
for e in range(experts):
    w = dequantize(packed[e], tensor.tensor_type).reshape(n,k).astype(np.float64)
    for x, ids, out in zip(inputs,routes,expected):
        token, slot = np.where(ids == e)
        if len(token): out[token,slot,:] = (x[token] @ w.T).astype('<f4')
for case, out in zip(cases,expected): out.tofile(args.output/case['expected'])
with args.model.open('rb') as stream: model_hash = hashlib.file_digest(stream,'sha256').hexdigest()
manifest={'model_sha256':model_hash,'format':'GGML Q4_K, 144 bytes/256 elements','seed':42,
          'reference':'Pinned GGUF dequantization, CPU NumPy f64 matmul, stored as f32',
          'max_relative_rmse':0.02,'max_normalized_error':0.05,
          'scope':'All 128 gate/up experts with resident top-8 IDs; includes GPU grouping and output scatter, excludes router logits/top-k selection, expert activation/down projection and combine',
          'cases':cases}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Exported',len(cases),'routed fixtures; weight slices remain local',flush=True)
