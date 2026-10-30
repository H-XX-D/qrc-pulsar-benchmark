"""Fetch the published Archibald CSV without replacing an existing local file."""

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

ZENODO_URL = "https://zenodo.org/records/1241122/files/tuned2_aligned_pulses_1400_nosun2_primary_residuals.csv?download=1"
ZENODO_MD5 = "f1f3094c8ce8245eff5cd2fec6e22f5d"


def ensure_zenodo_csv(path):
    """Download only when missing; check the publisher's checksum before use."""
    path = Path(path)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Expected a residual file, found a directory: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.md5()
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".zenodo-", delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(ZENODO_URL, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != ZENODO_MD5:
            raise ValueError("Downloaded Zenodo CSV checksum does not match the published release.")
        # Atomic creation fails instead of overwriting a file created concurrently.
        os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
