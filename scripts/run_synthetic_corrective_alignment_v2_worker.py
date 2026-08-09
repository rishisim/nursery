from __future__ import annotations

import argparse
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1]
while str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

from babyworld_lite.corrective_alignment_v2 import parallel as _parallel  # noqa: E402


EXPECTED_PARALLEL = SOURCE_ROOT / "babyworld_lite/corrective_alignment_v2/parallel.py"
if Path(str(_parallel.__file__)).resolve() != EXPECTED_PARALLEL.resolve():
    raise PermissionError("corrective worker imported parallel code outside SOURCE_ROOT")
_subprocess_worker_main = _parallel._subprocess_worker_main


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--request-manifest-sha256", required=True)
    arguments = parser.parse_args()
    return _subprocess_worker_main(
        arguments.request,
        arguments.request_sha256,
        arguments.request_manifest_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
