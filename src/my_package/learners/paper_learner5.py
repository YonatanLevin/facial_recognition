from my_package.learners.paper_learner1 import PaperLearner4

class PaperLearner5(PaperLearner4):
    def __init__(self, device):
        super().__init__(device, encoder_final_activation='Tanh')