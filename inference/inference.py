import torch
from models.model import UNet
from config import config  # CHANGE THIS: Remove 'config.config'

class CloudRemovalModel:
    def __init__(self, model_path=None):
        # Use the config singleton to get the device
        self.device = config.inference.device if torch.cuda.is_available() else 'cpu'
        
        # If model_path isn't provided, use the one from config
        if model_path is None:
            model_path = config.model.diffusion_weights
            
        self.model = UNet(in_channels=5, out_channels=4) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    # ... rest of your code
    def reconstruct(self, cloudy_image, mask):
        # 1. Convert numpy inputs to PyTorch Tensors
        # cloudy_image is assumed to be (C, H, W) numpy array
        # mask is assumed to be (H, W) or (1, H, W) numpy array
        
        input_tensor = torch.from_numpy(cloudy_image).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).float()
        
        # Ensure mask has (1, 1, H, W) shape for concatenation
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.to(self.device)

        # 2. Concatenate input image and mask to match UNet's 5-channel input
        # Based on your model.py, GeneratorInpainter expects (5, H, W)
        combined_input = torch.cat([input_tensor, mask_tensor], dim=1)

        # 3. Pass data through the model
        with torch.no_grad():
            output = self.model(combined_input)

        # 4. Convert back to numpy (C, H, W)
        reconstructed = output.squeeze(0).cpu().numpy()
        
        return reconstructed