from torch.nn import LeakyReLU

from my_package.learners.paper_learner4 import PaperLearner4

class PaperLearner8(PaperLearner4):
    def __init__(self, device):
        super().__init__(device, encoder_final_activation=LeakyReLU, 
                         encoder_conv_activation=LeakyReLU)