from torch.nn import LeakyReLU, Sigmoid

from my_package.learners.paper_learner4 import PaperLearner4

class PaperLearner9(PaperLearner4):
    def __init__(self, device):
        super().__init__(device, encoder_final_activation=Sigmoid, 
                         encoder_conv_activation=LeakyReLU)