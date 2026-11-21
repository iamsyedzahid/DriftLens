"""Download UCI Bank Marketing and record its provenance.

Usage: python scripts/downloadData.py
"""
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.dataLoader import computeChecksum

SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET_NAME = "bank-additional-full.csv"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def findMember(archive: zipfile.ZipFile, suffix: str) -> str:
    """Return the first archive member whose name ends with `suffix`."""
    matches = [n for n in archive.namelist() if n.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"{suffix} not found in archive")
    return matches[0]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(SOURCE_URL, timeout=120).read()))
    # The UCI archive nests bank-additional.zip inside the outer zip.
    inner = zipfile.ZipFile(io.BytesIO(outer.read(findMember(outer, "bank-additional.zip"))))
    targetPath = RAW_DIR / TARGET_NAME
    targetPath.write_bytes(inner.read(findMember(inner, TARGET_NAME)))
    namesMember = [n for n in inner.namelist() if n.endswith("bank-additional-names.txt")]
    _ = [(RAW_DIR / "bank-additional-names.txt").write_bytes(inner.read(n)) for n in namesMember[:1]]
    source = {"url": SOURCE_URL, "file": TARGET_NAME,
              "retrievedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "sha256": computeChecksum(str(targetPath))}
    (RAW_DIR / "SOURCE.json").write_text(json.dumps(source, indent=2))
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
