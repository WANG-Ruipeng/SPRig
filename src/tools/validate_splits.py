#!/usr/bin/env python3
"""Check three explicit sequence splits, optionally against parent-asset IDs."""
import argparse
import json
from pathlib import Path


def validate(train, validation, test, parent_map=None):
    groups={'train': train, 'validation': validation, 'test': test}
    problems=[]
    for name, rows in groups.items():
        if not rows: problems.append(f'{name}: empty split')
        if len(rows) != len(set(rows)): problems.append(f'{name}: duplicate sequence IDs')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if set(groups[a]) & set(groups[b]): problems.append(f'{a}/{b}: overlapping sequence IDs')
    if parent_map is not None:
        missing=set(sum(groups.values(),[]))-set(parent_map)
        if missing: problems.append(f'parent map missing {len(missing)} IDs')
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            pa={parent_map[x] for x in groups[a] if x in parent_map}
            pb={parent_map[x] for x in groups[b] if x in parent_map}
            if pa & pb: problems.append(f'{a}/{b}: overlapping parent assets')
    return {'valid': not problems, 'counts':{k:len(v) for k,v in groups.items()},
            'parent_assets_checked':parent_map is not None, 'problems':problems}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','validation','test'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-map',type=Path)
    a=p.parse_args()
    def read(path): return [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
    mapping=json.loads(a.parent_map.read_text()) if a.parent_map else None
    result=validate(read(a.train),read(a.validation),read(a.test),mapping)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__=='__main__': main()
