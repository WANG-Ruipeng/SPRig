"""Verify downloaded model bytes against the reviewed release hashes."""
import hashlib
from pathlib import Path


def verify_file(path, *, expected_size, expected_sha256):
    path = Path(path)
    size = path.stat().st_size
    if size != expected_size:
        raise ValueError(f"Wrong file size for {path.name}: {size} != {expected_size}")
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    if h.hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}; do not use this checkpoint.")
    return {'size': size, 'sha256': h.hexdigest()}
