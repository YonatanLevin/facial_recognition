import torch
import torch.nn.init as init

from comparison_heads.comparison_head import ComparisonHead

class PaperHead(ComparisonHead):
    def __init__(self, encoding_dim):
        super().__init__(encoding_dim)

        self.linear_weight_mean = 0
        self.linear_weight_std = 2*10**-1 

        self.linear_bias_mean = 0.5
        self.linear_bias_std = 10**-2  

        self.fc = torch.nn.Linear(self.encoding_dim, out_features=1, bias=True)
        init.normal_(self.fc.weight, self.linear_weight_mean, self.linear_weight_std)
        init.normal_(self.fc.bias, self.linear_bias_mean, self.linear_bias_std)

    def forward(self, img1, img2):
        component_dist = torch.abs(img1 - img2)
        logits = self.fc(component_dist)
        return logits
    
    def logits_to_probs(self, logits):
        return torch.sigmoid(logits)
    