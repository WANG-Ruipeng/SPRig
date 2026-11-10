#!/usr/bin/env python3
"""Build an archive only from reviewed source paths with matching SHA-256 values."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ALLOWED_SUFFIXES={'.py','.md','.txt','.json'}


def reviewed_files(root):
    root=Path(root).resolve()
    manifest=root/'RELEASE_MANIFEST.json'
    records=json.loads(manifest.read_text())['files']
    selected=[]; seen=set()
    for record in records:
        rel=record['path']; parts=PurePosixPath(rel)
        if parts.is_absolute() or '..' in parts.parts or '\\' in rel or rel in seen:
            raise ValueError(f'Invalid or duplicate path: {rel}')
        seen.add(rel)
        path=root/rel
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            raise ValueError(f'Invalid file: {rel}')
        if path.suffix not in ALLOWED_SUFFIXES and rel!='.gitignore':
            raise ValueError(f'Unreviewed format: {rel}')
        if path.stat().st_size > 2*1024*1024:
            raise ValueError(f'Unexpected source size: {rel}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Source changed since review: {rel}')
        selected.append(path)
    # Unknown files are intentionally never swept into the archive.
    return selected+[manifest]


def build(root, output):
    root=Path(root).resolve();output=Path(output).resolve()
    files=reviewed_files(root)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,'SPRig/'+str(path.relative_to(root)))
    return len(files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); n=build(a.root,a.output)
    print(json.dumps({'output':str(a.output.resolve()),'files':n,'data_included':False}))


if __name__=='__main__': main()
