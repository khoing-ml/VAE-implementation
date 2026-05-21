import os
import random

try:
    import numpy as np
except ImportError:
    np = None


config = {
    "name": "VAE",
    "in_channels": 3,
    "latent_dim": 128,
    "data_path": "data/",
    "train_batch_size": 64,
    "val_batch_size": 64,
    "patch_size": 64,
    "num_workers": 4,
    "lr": 0.005,
    "weight_decay": 0.0,
    "scheduler_gamma": 0.95,
    "kld_weight": 0.00025,
    "manual_seed": 1265,
    "gpus": [0, 1],
    "max_epochs": 100,
    "save_dirs": "logs/",
    "eval_recon_metrics_every": 1,
    "eval_recon_metrics_num_batches": 20,
    "eval_fid_every": 5,
    "fid_num_samples": 1000,
}

if config.get("gpus"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config["gpus"])


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