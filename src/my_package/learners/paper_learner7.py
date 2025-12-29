from torch.nn import ReLU

from my_package.learners.paper_learner4 import PaperLearner4

class PaperLearner7(PaperLearner4):
    def __init__(self, device):
        super().__init__(device, encoder_final_activation=ReLU, encoder_conv_activation=ReLU)