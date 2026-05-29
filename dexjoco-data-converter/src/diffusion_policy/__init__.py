# this is a wraper

import sys
from pathlib import Path

_LOCAL_PACKAGE_DIR = Path(__file__).resolve().parent
_THIRD_PARTY_ROOT = _LOCAL_PACKAGE_DIR.parent / "3rd" / "diffusion_policy"
_THIRD_PARTY_PACKAGE_DIR = _THIRD_PARTY_ROOT / "diffusion_policy"

third_party_root_str = str(_THIRD_PARTY_ROOT)
if third_party_root_str not in sys.path:
    sys.path.insert(0, third_party_root_str)

__path__ = [str(_LOCAL_PACKAGE_DIR), str(_THIRD_PARTY_PACKAGE_DIR)]
