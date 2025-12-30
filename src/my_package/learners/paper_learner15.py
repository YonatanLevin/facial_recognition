from torch.nn import LeakyReLU, Sigmoid

from my_package.learners.paper_learner4 import PaperLearner4

class PaperLearner15(PaperLearner4):
    def __init__(self, device):
        super().__init__(device, encoder_final_activation=Sigmoid, 
                         encoder_conv_activation=LeakyReLU, 
                         encoder_linear_batch_norm=True,
                         encoder_conv_batch_norm=True,
                         normalize_imgs=True,
                         l1_lambda=1e-3)