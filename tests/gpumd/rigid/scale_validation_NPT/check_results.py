"""Recheck existing NPT output; does not run MD or the static NEP simulations."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scale_validation_NVT'))
from check import verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    if json.loads((args.folder / 'manifest.json').read_text())['ensemble'] != 'rigid_npt':
        parser.error('Expected rigid_npt manifest')
    raise SystemExit(0 if verify(args.folder) else 1)
