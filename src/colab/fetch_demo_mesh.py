"""Fetch Puppeteer's pinned example mesh without redistributing it in SPRig."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def matches(path, spec):
    return (path.is_file() and path.stat().st_size == spec["size"]
            and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"])


def fetch_mesh(output):
    spec = json.loads((Path(__file__).parent / "runtime-lock.json").read_text())["demo"]
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not matches(output, spec):
        temporary = None
        try:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "SPRig-Colab-Demo"})
            with urllib.request.urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    size = 0
                    while block := response.read(64 * 1024):
                        size += len(block)
                        if size > spec["size"]:
                            raise ValueError("Example download exceeds its pinned size.")
                        stream.write(block)
            if not matches(temporary, spec):
                raise ValueError("Puppeteer example checksum mismatch; inference was not started.")
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"mesh": str(output), **spec}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch_mesh(args.output), indent=2))


if __name__ == "__main__":
    main()
