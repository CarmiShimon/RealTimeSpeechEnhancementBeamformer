import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import os
from hydra.core.hydra_config import HydraConfig
# Import your cleaned classes
from dataset import SpatialDataset
from trainer import Trainer
from model import SpatialMaskNet  # Replace with your actual model class name


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Log the configuration (Great for tracking experiments)
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------")
    hydra_run_dir = HydraConfig.get().run.dir
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Initialize Model
    # Passing the specific model sub-config (e.g., hidden_dim, layers)
    model = SpatialMaskNet(cfg, device)

    # 4. Initialize DataLoaders
    train_ds = SpatialDataset(cfg, split='train')
    val_ds = SpatialDataset(cfg, split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.get('num_workers', 4)
    )

    # 5. Initialize Trainer
    # This trainer now contains the STFTHandler and MVDRBeamformer internally
    trainer = Trainer(cfg, model, device)
    best_val_loss = float('inf')
    patience = 4
    counter = 0
    # 6. Training Loop
    print(f"Starting training for {cfg.training.epochs} epochs...")
    for epoch in range(cfg.training.epochs):
        avg_train_loss = trainer.train_epoch(train_loader, epoch)
        # Validate
        avg_val_loss = trainer.validate(val_loader)
        print(f"===> Epoch {epoch} Complete. Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")
        # Save best model logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # reset counter
            # Save the model weights
            checkpoint_path = os.path.join(hydra_run_dir, f"model_epoch_{epoch + 1}.pt")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            # torch.save(model.state_dict(), checkpoint_path)

            print(f"--- Saved Best Model at Epoch {epoch} ---")
        else:
            print(f"Validation loss did not improve from {best_val_loss:.4f}")
            counter += 1
            if counter >= patience:
                print("Early stopping triggered. Training finished.")
                break



if __name__ == "__main__":
    main()