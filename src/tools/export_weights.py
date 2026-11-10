#!/usr/bin/env python3
"""Export explicitly chosen checkpoints as model tensors only; requires PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def select_state_dict(package, torch):
    if not isinstance(package, dict) or not package:
        raise ValueError('Expected a non-empty state_dict or checkpoint mapping.')
    if isinstance(package.get('model'), dict):
        state = package['model']
    elif isinstance(package.get('state_dict'), dict):
        state = package['state_dict']
    else:
        state = package
    if not state or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items()):
        raise ValueError('Model state must contain only string keys and tensors.')
    # Do not rewrite prefixes or shapes: that can mask an architecture mismatch.
    return {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}


def export_one(source, destination, stage):
    import torch
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    package = torch.load(source, map_location='cpu', weights_only=True)
    state = select_state_dict(package, torch)
    schema = [{'name': k, 'shape': list(v.shape), 'dtype': str(v.dtype), 'numel': v.numel()}
              for k, v in state.items()]
    del package
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.sprig-export-', suffix='.pth', dir=destination.parent)
    os.close(fd)
    try:
        torch.save({'model': state}, temp)
        check = torch.load(temp, map_location='cpu', weights_only=True)
        if set(check) != {'model'} or list(check['model']) != list(state):
            raise RuntimeError('Export key verification failed.')
        for k, v in state.items():
            w = check['model'][k]
            # Byte equality handles even identical NaN payloads without changing tensors.
            if v.dtype != w.dtype or v.shape != w.shape or not torch.equal(
                v.reshape(-1).view(torch.uint8), w.reshape(-1).view(torch.uint8)
            ):
                raise RuntimeError(f'Tensor verification failed: {k}')
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temp, destination)
    finally:
        Path(temp).unlink(missing_ok=True)
    return {'stage': stage, 'file': destination.name, 'bytes': destination.stat().st_size,
            'sha256': sha256(destination), 'source_sha256': sha256(source),
            'tensor_count': len(state), 'parameter_and_buffer_elements': sum(v.numel() for v in state.values()),
            'tensor_schema': schema, 'runtime_compatibility_verified': False,
            'training_dataset': 'DT4D (owner supplied description; run lineage not certified)',
            'payload': 'model mapping only; no optimizer, training arguments, or dataset examples'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skeleton', type=Path, required=True)
    parser.add_argument('--skinning', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent directory')
    args = parser.parse_args()
    for source in (args.skeleton, args.skinning):
        if not source.is_file(): parser.error(f'Checkpoint does not exist: {source}')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for stage, source in [('skeleton', args.skeleton), ('skinning', args.skinning)]:
        records.append(export_one(source, args.output/f'sprig_dt4d_{stage}.pth', stage))
    (args.output/'MODEL_MANIFEST.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.output/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in records))
    (args.output/'MODEL_CARD.md').write_text('''# SPRig DT4D model weights

These files contain only neural-network model tensors. They contain no distributed
DT4D mesh assets, per-mesh labels, optimizer states, or training-path arguments.
The author selected DeformingThings4D_finetune_v2 for skeleton and
finetuned_model_v3 for skinning. Supply those two checkpoints to this exporter.
The exporter verifies tensor equality; it does not perform model inference.
Add the model license and the actual A100 validation result before public release.
The full tensor schema and source/output SHA-256 checksums are in MODEL_MANIFEST.json.
''')
    print(json.dumps([{'stage':r['stage'],'file':r['file'],'bytes':r['bytes'],'sha256':r['sha256']} for r in records],indent=2))


if __name__ == '__main__':
    main()
