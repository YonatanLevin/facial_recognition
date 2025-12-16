import torch

from comparison_heads.comparison_head import ComparisonHead

class PaperHead(ComparisonHead):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)
        self.fc = torch.nn.Linear(in_features=hyper_parameters['encoding_dim'], out_features=1, bias=False)

    def forward(self, img1, img2, is_probs):
        component_dist = torch.abs(img1 - img2)
        logits = self.fc(component_dist)
        if is_probs:
            return torch.sigmoid(logits)
        return logits
