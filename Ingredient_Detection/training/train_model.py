import subprocess
import argparse

NUM_FOLDS = 5

parser = argparse.ArgumentParser(description="YOLO K-fold cross validation")
parser.add_argument("--model-size", type=str, required=True, choices=["n", "s", "m", "l", "x"])
parser.add_argument("--batch-size", type=int, required=True)

args = parser.parse_args()

for i in range(0,NUM_FOLDS):
  cmd = [
    "python3",
    "train_model_fold.py",
    "--model-size", args.model_size,
    "--batch-size", str(args.batch_size),
    "--iter",str(i)
  ]
  try:
    subprocess.run(cmd, check=True)
    print(f"Completed computations for iter{i}")
  except:
    print(f"Error in k-fold cross validation: {args.model_size}")
