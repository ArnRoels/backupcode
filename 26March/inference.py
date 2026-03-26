from train import classify_and_move_images
from model import QualityModel
import torch
from config import Config
from torchvision import transforms
from remove_empty import remove_empty_bad_quality_folders

base_path = 'I:/Export Bart/Stationary_phase_screen'
plate_ids = ['89B', 'AA', 'AB', 'B']  # Add more plate IDs as needed

ev_dirs = [f"{base_path}/PLATE{plate_id}" for plate_id in plate_ids]

# Create transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

# Load model
model = QualityModel().to(Config.DEVICE)
print('Device:', Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_PATH))

# Loop through each directory
for ev_dir in ev_dirs:
    print(f"Processing directory: {ev_dir}")
    classify_and_move_images(ev_dir, model, transform)
    remove_empty_bad_quality_folders(ev_dir)
    print(f"Finished processing: {ev_dir}")