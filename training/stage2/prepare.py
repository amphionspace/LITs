"""Prepare the current audited Stage 2 joint dataset and initialization."""
import hashlib


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    from training.majestic_scratch.prepare import prepare
    raise SystemExit(prepare())
