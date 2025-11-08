import wandb
from pathlib import Path

from model.model import NormalizedRegressionMLP
from model.utils import train


def main():
    run = wandb.init(project="zneus_part_2")
    config = wandb.config
    run.name = f"lr_{config.learning_rate}_bs_{config.batch_size}_hu_{config.hidden_units}_{config.optimizer}_{config.loss_function}"

    _, history, test_loss = train(
        data_path=Path("houses.csv"),
        model=NormalizedRegressionMLP,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        epochs=config.epochs,
        batch_size=config.batch_size,
        hidden_units=config.hidden_units,
        learning_rate=config.learning_rate,
        early_stopping=config.early_stopping,
        loss_function=config.loss_function,
        optimizer=config.optimizer,
    )

    wandb.log(
        {
            "train_loss": history["train_loss"][-1],
            "test_loss": test_loss,
            "val_loss": history["val_loss"][-1],
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
