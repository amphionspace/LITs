"""Public entry point for the current Stage 2 joint preflight."""
import argparse
from pathlib import Path
from training.majestic_scratch.preflight import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.run_dir)
