import torch
import random
import torchvision.transforms.v2.functional as F
from databases.img_transformer import ImgTransformer

class AffineTransformer(ImgTransformer):
    def __init__(self):
        super().__init__()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Define the potential transformation parameters
        # In a real "paper" implementation, we sample these ranges
        angle = random.uniform(-10, 10)
        translations = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        scale = random.uniform(0.8, 1.2)
        shear = random.uniform(-0.3, 0.3)

        # Apply each component with 0.5 probability
        if random.random() < 0.5:
            # Apply rotation
            img = F.rotate(img, angle=angle)
            
        if random.random() < 0.5:
            # Apply translation and scale (often grouped in affine)
            img = F.affine(img, angle=0, translate=translations, scale=scale, shear=0)
            
        if random.random() < 0.5:
            # Apply shear
            img = F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=shear)

        return img