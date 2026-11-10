#!/usr/bin/env python3
"""Synthetic checkpoint round-trip test; requires PyTorch; no dataset or GPU needed."""
import tempfile
from pathlib import Path
import torch
from export_weights import export_one

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    expected={'linear.weight':torch.arange(12,dtype=torch.float32).reshape(3,4),
              'counter':torch.tensor(3,dtype=torch.int64)}
    src=root/'training.pth';dst=root/'inference.pth'
    torch.save({'model':expected,'optimizer':{'private_state':torch.ones(2)},
                'args':{'data_root':'private-test-path'},'epoch':4},src)
    report=export_one(src,dst,'synthetic')
    actual=torch.load(dst,map_location='cpu',weights_only=True)
    assert set(actual)=={'model'}
    assert all(torch.equal(actual['model'][k],v) for k,v in expected.items())
    assert report['tensor_count']==2
    assert 'private-test-path' not in str(report)
    print('PASS: tensor equality and non-model field exclusion')
