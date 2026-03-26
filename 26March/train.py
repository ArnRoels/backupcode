from dataset import ImageDataset
from config import Config
from model import QualityModel, Trainer, get_data_loaders
import torch.optim as optim
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


def main():
    train_losses = []
    val_losses = []
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    print(f"✅ Training data: {len(train_loader.dataset)} images, "
          f"Validation data: {len(val_loader.dataset)} images")

    # Initialize model and training components
    model = QualityModel().to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, Config.DEVICE)
    
    for epoch in range(Config.EPOCHS):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate()
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            trainer.early_stop_count = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("✅ Model improved & saved!")
        else:
            trainer.early_stop_count += 1
            print(f"Early stopping counter: {trainer.early_stop_count}/{Config.PATIENCE}")
            
        if trainer.early_stop_count >= Config.PATIENCE:
            print("⏹️ Early stopping triggered!")
            break
    
    # Visualization of loss curves after training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()

# inference.py
import shutil
from torch.utils.data import DataLoader
def classify_and_move_images(root_folder, model, transform):
    model.eval()
    
    # Get subfolders once
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)
        bad_quality_folder = f"{subfolder_path}_bad_Q"
        os.makedirs(bad_quality_folder, exist_ok=True)

        # Create dataset and DataLoader (minimal memory footprint with batch_size=1)
        dataset = ImageDataset(subfolder_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)  # num_workers=0 to avoid memory overhead

        for img_tensor, _, filename in dataloader:
            # Only evaluate images ending with _C1.tiff
            if not filename[0].endswith("_C1.tiff"):
                continue

            img_tensor = img_tensor.to(Config.DEVICE)
            with torch.no_grad():
                prediction = model(img_tensor).item()

            # Explicitly delete tensor to free memory
            del img_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move files if low quality
            if prediction < 0.5:
                base = filename[0].replace("_C1.tiff", "")
                for suffix in ["_C1.tiff", "_C2.tiff", "_C3.tiff", "_C4.tiff", "_C5.tiff"]:
                    target_filename = base + suffix
                    source_path = os.path.join(subfolder_path, target_filename)
                    if os.path.exists(source_path):
                        shutil.move(source_path, os.path.join(bad_quality_folder, target_filename))
                        print(f"Moved {target_filename} to {bad_quality_folder}")

        print(f"Processed folder: {subfolder_path}")

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()