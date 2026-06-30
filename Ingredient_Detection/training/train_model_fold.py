from ultralytics import YOLO
from pathlib import Path
import argparse

EPOCHS = 100
NUM_ITER = 5
SEED = 42


parser = argparse.ArgumentParser(description="Train YOLO model with specific folds of iteration")
parser.add_argument("--model-size", type=str, required=True, choices=["n", "s", "m", "l", "x"])
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--iter", type=int, required=True, choices=range(0, NUM_ITER))


args = parser.parse_args()
current_dir = Path.cwd()


fold_dir_name = f"dataset_kfold/iter_{args.iter}"
project_dir = f"res/model_{args.model_size}"

fold_config_path = current_dir / fold_dir_name / f"data_iter_{args.iter}.yaml"
project_path = current_dir / project_dir

print(f"Starting training for yolo11{args.model_size}")

model = YOLO(f"yolo11{args.model_size}.pt", task="detect")
model.train(
    data=fold_config_path,
    project=project_path,
    name= f"iter_{args.iter}",
    device="cuda",
    epochs=EPOCHS,
    seed=SEED,
    batch=args.batch_size ,
    scale=0.2, 
    val= True
)
