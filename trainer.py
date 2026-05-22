import os
import wandb
import random
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


class VAETrainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 config,
                 device
                 ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.config["weight_decay"]
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config["scheduler_gamma"]
        )

        self.best_val_loss = float("inf")
        self.global_step = 0
        os.makedirs(config["save_dirs"], exist_ok=True)

    def get_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recons_loss = 0.0
        total_kld_loss = 0.0

        progress_bar = tqdm(self.train_loader,  desc=f"Epoch {epoch + 1}/{self.config['max_epochs']} [Train]")

        for batch_idx , batch in enumerate(progress_bar):
            x , _ = batch
            x = x.to(self.device)
            self.optimizer.zero_grad()
            recons, input_x, mu, log_var = self.model(x)
            vae_model = self.get_model()
            loss_dict = vae_model.loss_function(
                recons,
                input_x,
                mu,
                log_var,
                M_N=self.config["kld_weight"],
            )
            loss = loss_dict["loss"]
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            recons_loss = loss_dict["Reconstruction_Loss"]
            kld_loss = loss_dict["KLD"]

            total_loss += loss.item()
            total_recons_loss += recons_loss.item()
            total_kld_loss += kld_loss.item()

            current_lr = self.optimizer.param_groups[0]["lr"]

            wandb.log({
                "train/step_loss": loss.item(),
                "train/recons_loss": recons_loss.item(),
                "train/kld_loss": kld_loss.item(),
                "train/lr": current_lr,
                "epoch": epoch + 1,
            }, step = self.global_step
            )
            
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "recons_loss": recons_loss.item(),
                    "kld_loss": kld_loss.item(),
                    "lr": current_lr,
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_recons_loss = total_recons_loss / len(self.train_loader)
        avg_kld_loss = total_kld_loss / len(self.train_loader)

        return {
            "loss": avg_loss,
            "reconstruction_loss": avg_recons_loss,
            "kld_loss": avg_kld_loss,
        }

    def to_01(self, x):
        image_range = self.config.get("image_range", "0_1")
        if image_range == "-1_1":
            x = (x + 1.0) / 2.0
        return torch.clamp(x, 0.0, 1.0)
    
    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recons_loss = 0.0
        total_kld_loss = 0.0

        progress_bar = tqdm(self.val_loader,  desc=f"Epoch {epoch + 1}/{self.config['max_epochs']} [Train]")

        for batch_idx , batch in enumerate(progress_bar):
            x , _ = batch
            x = x.to(self.device)
            recons, input_x, mu, log_var = self.model(x)
            vae_model = self.get_model()
            loss_dict = vae_model.loss_function(
                recons,
                input_x,
                mu,
                log_var,
                M_N=self.config["kld_weight"],
            )
            loss = loss_dict["loss"]
            recons_loss = loss_dict["Reconstruction_Loss"]
            kld_loss = loss_dict["KLD"]

            total_loss += loss.item()
            total_recons_loss += recons_loss.item()
            total_kld_loss += kld_loss.item()
            progress_bar.set_postfix(
                {
                    "val_loss": loss.item(),
                    "recons": recons_loss.item(),
                    "kld": kld_loss.item(),
                }
            )

        avg_loss = total_loss / len(self.val_loader)
        avg_recons_loss = total_recons_loss / len(self.val_loader)
        avg_kld_loss = total_kld_loss / len(self.val_loader)

        return {
            "loss": avg_loss,
            "reconstruction_loss": avg_recons_loss,
            "kld_loss": avg_kld_loss,
        }

    @torch.no_grad()
    def log_reconstructions(self,epoch):
        self.model.eval()

        batch = next(iter(self.val_loader))
        x, _ = batch
        x = x.to(self.device)
        vae_model = self.get_model()
        recons = vae_model.generate(x)

        comparison = torch.cat([
            x[:8],
            recons[:8],
        ])

        grid = make_grid(
            comparison.cpu(),
            nrow=8,
            normalize=True,
        )

        wandb.log({
            "images/reconstructions": wandb.Image(
                grid,
                caption=f"Epoch {epoch + 1}: top=original, bottom=reconstruction"
            )
        }, step=self.global_step)

    @torch.no_grad()
    def log_samples(self, epoch, num_samples=16):
        self.model.eval()
        vae_model = self.get_model()
        samples = vae_model.sample(
            num_samples = num_samples,
            current_device = self.device,
        )
        grid = make_grid(
            samples.cpu(),
            nrow=4,
            normalize=True,
        )

        wandb.log({
            "images/samples": wandb.Image(
                grid,
                caption=f"Epoch {epoch + 1}: Generated samples"
            )
        }, step=self.global_step)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        vae_model = self.get_model()
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": vae_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "global_step": self.global_step,
        }

        last_path = os.path.join(
            self.config["save_dirs"],
            "last.ckpt"
        )

        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(
                self.config["save_dirs"],
                "best.ckpt"
            )
            torch.save(checkpoint, best_path)

            wandb.save(best_path)

        wandb.save(last_path)

    @torch.no_grad()
    def evaluate_reconstruction_metrics(self, epoch):
        self.model.eval()
    
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
    
        # LPIPS with normalize=True expects images in [0, 1]
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True
        ).to(self.device)
    
        max_batches = self.config.get("eval_recon_metrics_num_batches", 20)
    
        for batch_idx, batch in enumerate(
            tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Recon Metrics]")
        ):
            x, _ = batch
            x = x.to(self.device)
    
            recons, input_x, mu, log_var = self.model(x)
    
            x_01 = self.to_01(input_x)
            recons_01 = self.to_01(recons)
    
            psnr_metric.update(recons_01, x_01)
            ssim_metric.update(recons_01, x_01)
            lpips_metric.update(recons_01, x_01)
    
            if batch_idx + 1 >= max_batches:
                break
    
        psnr = psnr_metric.compute().item()
        ssim = ssim_metric.compute().item()
        lpips_score = lpips_metric.compute().item()
    
        wandb.log({
            "val/psnr": psnr,
            "val/ssim": ssim,
            "val/lpips": lpips_score,
            "epoch": epoch + 1,
        }, step=self.global_step)
    
        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_score,
        }

    @torch.no_grad()
    def evaluate_rfid(self, epoch):
        self.model.eval()
    
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True
        ).to(self.device)
    
        max_samples = self.config.get("fid_num_samples", 2048)
        num_seen = 0
    
        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [rFID]"):
            x, _ = batch
            x = x.to(self.device)
    
            recons, input_x, mu, log_var = self.model(x)
    
            real_imgs = self.to_01(input_x)
            recon_imgs = self.to_01(recons)
    
            fid.update(real_imgs, real=True)
            fid.update(recon_imgs, real=False)
    
            num_seen += x.size(0)
            if num_seen >= max_samples:
                break
    
        rfid = fid.compute().item()
    
        wandb.log({
            "val/rfid": rfid,
            "epoch": epoch + 1,
        }, step=self.global_step)
    
        return rfid

    @torch.no_grad()
    def evaluate_gfid(self, epoch):
        self.model.eval()
        vae_model = self.get_model()
    
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True
        ).to(self.device)
    
        max_samples = self.config.get("fid_num_samples", 2048)
        num_seen = 0
    
        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [gFID]"):
            real_imgs, _ = batch
            real_imgs = real_imgs.to(self.device)
    
            batch_size = real_imgs.size(0)
    
            fake_imgs = vae_model.sample(
                num_samples=batch_size,
                current_device=self.device
            )
    
            real_imgs = self.to_01(real_imgs)
            fake_imgs = self.to_01(fake_imgs)
    
            fid.update(real_imgs, real=True)
            fid.update(fake_imgs, real=False)
    
            num_seen += batch_size
            if num_seen >= max_samples:
                break
    
        gfid = fid.compute().item()
    
        wandb.log({
            "val/gfid": gfid,
            "epoch": epoch + 1,
        }, step=self.global_step)
    
        return gfid
    def fit(self):
        for epoch in range(self.config["max_epochs"]):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            wandb.log({
                "epoch": epoch + 1,

                "train/loss": train_metrics["loss"],
                "train/reconstruction_loss": train_metrics["reconstruction_loss"],
                "train/kld_loss": train_metrics["kld_loss"],

                "val/loss": val_metrics["loss"],
                "val/reconstruction_loss": val_metrics["reconstruction_loss"],
                "val/kld_loss": val_metrics["kld_loss"],

                "lr": current_lr,
            }, step=self.global_step)

            self.log_reconstructions(epoch)
            self.log_samples(epoch)

            recon_metric_results = None
            rfid_score = None
            gfid_score = None
    
            eval_recon_every = self.config.get("eval_recon_metrics_every", 1)
            eval_fid_every = self.config.get("eval_fid_every", 5)
    
            if (epoch + 1) % eval_recon_every == 0:
                recon_metric_results = self.evaluate_reconstruction_metrics(epoch)
    
            if (epoch + 1) % eval_fid_every == 0:
                rfid_score = self.evaluate_rfid(epoch)
                gfid_score = self.evaluate_gfid(epoch)
            val_loss = val_metrics["loss"]

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
    
            self.save_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                is_best=is_best,
            )
    
            msg = (
                f"Epoch [{epoch + 1}/{self.config['max_epochs']}] "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Best Val Loss: {self.best_val_loss:.4f}"
            )
    
            if recon_metric_results is not None:
                msg += (
                    f" | PSNR: {recon_metric_results['psnr']:.4f}"
                    f" | SSIM: {recon_metric_results['ssim']:.4f}"
                    f" | LPIPS: {recon_metric_results['lpips']:.4f}"
                )
    
            if rfid_score is not None:
                msg += f" | rFID: {rfid_score:.4f}"
    
            if gfid_score is not None:
                msg += f" | gFID: {gfid_score:.4f}"
    
            print(msg)