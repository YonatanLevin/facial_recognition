import random

import numpy as np
import torch


from my_package.pipeline import Pipeline

def main():
    set_seed()
    pipeline = Pipeline()
    pipeline.execute_learner()
    pipeline.print_best_learner_metrics()

def set_seed(seed=42):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # Set seed for torch (both CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()