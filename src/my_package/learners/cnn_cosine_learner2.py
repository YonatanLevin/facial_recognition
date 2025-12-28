
import torch

from my_package.comparison_heads.cosine_head import CosineHead
from my_package.learners.cnn_cosine_learner1 import CNNCosineLearner1
from my_package.learners.learner import Learner
from my_package.model import Model
from my_package.siamese_encoders.paper_cnn import PaperCNN


class CNNCosineLearner2(CNNCosineLearner1):
    def __init__(self, device, use_foreground: bool=False):
        super().__init__(device, use_foreground, encoder_final_activation='Tanh')
