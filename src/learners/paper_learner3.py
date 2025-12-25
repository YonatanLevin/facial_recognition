from learners.paper_learner1 import PaperLearner1

class PaperLearner3(PaperLearner1):
    def __init__(self, device):
        super().__init__(device, resize_size=(105,105), use_foreground=True)