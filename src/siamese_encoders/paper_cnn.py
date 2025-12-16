from torch.nn import Sequential, ReLU, MaxPool2d, AdaptiveMaxPool2d, Conv2d, Linear, Sigmoid, Flatten

from encoder import Encoder

class PaperCNN(Encoder):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)        
        
        self.network = Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=10),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            ReLU(),
            AdaptiveMaxPool2d(output_size=(6, 6)),
            
            Flatten(),
            Linear(in_features=9216, out_features=4096),
            Sigmoid()
        )

    def forward(self, img):
        return self.network(img)