from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def main():
    parser = argparse.ArgumentParser(description="kessler shield pipeline runner")
    parser.add_argument(
        "stage",
        choices=["train", "predict", "evaluate", "all"],
        help="terminal command *python main.py operation*",
    )

    args = parser.parse_args()

    if args.stage == "train":
        from training import training_pipeline
        training_pipeline()

    elif args.stage == "predict":
        from predict import predicting_pipeline
        predicting_pipeline()

    elif args.stage == "evaluate":
        from evaluation import evaluation_pipeline
        evaluation_pipeline()

    elif args.stage == "all":
        from training import training_pipeline
        from evaluation import evaluation_pipeline

        training_pipeline()
        print("[1/2] training complete")

        evaluation_pipeline()
        print("[2/2] evaluation complete")

if __name__ == "__main__":
    main()