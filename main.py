import argparse
import sys
from model.utils import train, plot_loss_curves
from pathlib import Path

from tools import load_models


def train_command(args: argparse.Namespace) -> None:
    """
    Placeholder training function.
    Replace this with your actual ML training logic.
    """
    df_path = Path(args.dataset)
    assert df_path.exists(), "Please provide a valid file path for the dataset."

    train_ratio, val_ratio, test_ratio = args.split_ratio
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1."

    models_dict = load_models("model.model")
    assert (
        args.model in models_dict
    ), f"Model not found in available models: {list(models_dict.keys())}"

    mod, hist, tst_ls = train(
        df_path,
        models_dict[args.model],
        train_ratio,
        val_ratio,
        test_ratio,
        args.epochs,
        args.batch_size,
        args.hidden_units,
        args.learning_rate,
        args.early_stopping,
        args.loss_function,
        args.optimizer,
        args.wandb_name,
        args.wandb,
        debug=True,
    )

    print("Training complete!")
    plot_loss_curves(hist)


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the training utility."""
    parser = argparse.ArgumentParser(
        description="Neural network training utility with configurable parameters."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train a model on a dataset")

    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Relative path to the dataset file or folder",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="RegressionMLP",
        help="Model architecture to use (default: RegressionMLP; check model/model.py for options)",
    )
    train_parser.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios for train/validation/test sets (default: 0.6 0.2 0.2)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    train_parser.add_argument(
        "--hidden-units",
        type=int,
        default=64,
        help="Number of hidden units in the model (default: 64)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    train_parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping (default: False)",
    )
    train_parser.add_argument(
        "--loss-function",
        type=str,
        default="MAE",
        choices=["MAE", "MSE"],
        help="Loss function to use (default: MAE)",
    )
    train_parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="Optimizer to use (default: Adam)",
    )
    train_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: True)",
    )
    train_parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional custom W&B run name",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    else:
        print("Unknown command:", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
