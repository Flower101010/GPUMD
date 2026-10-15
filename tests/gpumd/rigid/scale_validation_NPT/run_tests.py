"""NPT entry point; reuse the NVT fixture, execution and structural validation."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scale_validation_NVT'))
from run_tests import main

if __name__ == '__main__':
    main(ensemble='rigid_npt', suite_dir=HERE)
