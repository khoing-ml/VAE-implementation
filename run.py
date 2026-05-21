import os
import random
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "vae.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load config/vae.yaml. Install it with: pip install PyYAML"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    try:
        import torch
        import wandb

        from dataset import VAEDataset
        from model.vae import VAE
        from trainer import VAETrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install torch, wandb, torchvision, and torchmetrics before running."
        ) from exc

    config = load_config()

    if config.get("gpus"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config["gpus"])

    os.makedirs(config["save_dirs"], exist_ok=True)

    seed_everything(config["manual_seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", num_gpus)

    wandb.init(
        project="vae-training",
        name=config["name"],
        config=config,
    )

    model = VAE(
        in_channels=config["in_channels"],
        latent_dim=config["latent_dim"],
    )

    if torch.cuda.is_available() and num_gpus > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    data_module = VAEDataset(
        data_path=config["data_path"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        patch_size=config["patch_size"],
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        dataset_type=config.get("dataset_type", "celeba"),
        mnist_download=config.get("mnist_download", True),
    )
    data_module.setup()

    trainer = VAETrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        config=config,
        device=device,
    )

    trainer.fit()
    wandb.finish()


if __name__ == "__main__":
    main()