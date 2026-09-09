import argparse
import random

SHUFFLE_SEED = 42


parser = argparse.ArgumentParser()
parser.add_argument("--input_txt", type=str, required=True)
parser.add_argument("--output_txt", type=str, required=False)
parser.add_argument("--multiplier", type=int, default=1)
args = parser.parse_args()

INPUT_TXT = args.input_txt
if not args.output_txt:
    OUTPUT_TXT = INPUT_TXT.replace(".txt", f"_x{args.multiplier}.txt")
else:
    OUTPUT_TXT = args.output_txt
with open(INPUT_TXT, "r") as f:
    original_lines = f.readlines()

oversampled_lines = original_lines * args.multiplier
random.seed(SHUFFLE_SEED)
random.shuffle(oversampled_lines)

with open(OUTPUT_TXT, "w") as f:
    f.writelines(oversampled_lines)

