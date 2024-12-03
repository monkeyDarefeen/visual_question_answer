import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# Function to load the SAM model
def load_model(checkpoint_path, model_type='vit_h'):
    """
    Load the SAM model based on the checkpoint and model type.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    return mask_generator, device, sam