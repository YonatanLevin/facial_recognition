from torch.nn import Sequential, MaxPool2d, AdaptiveMaxPool2d, Conv2d, Linear, Sigmoid, \
                     Flatten, Module, BatchNorm1d, BatchNorm2d, Identity
import torch.nn.init as init

from my_package.siamese_encoders.encoder import Encoder

class PaperCNN(Encoder):
    def __init__(self, final_activation_class: Module | None = Sigmoid, conv_activation_class = Sigmoid,
                 linear_bias_mean = 0.5, linear_batch_norm = False, conv_batch_norm = False):
        super().__init__(encoding_dim=4096)

        self.conv_weight_mean = 0
        self.conv_weight_std = 10**-2  

        self.conv_bias_mean = 0.5
        self.conv_bias_std = 10**-2  

        self.linear_weight_mean = 0
        self.linear_weight_std = 2*10**-1 

        self.linear_bias_mean = linear_bias_mean
        self.linear_bias_std = 10**-2

        
        self.network = Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=10),
            BatchNorm2d(64) if conv_batch_norm else Identity(),
            conv_activation_class(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            BatchNorm2d(128) if conv_batch_norm else Identity(),
            conv_activation_class(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            BatchNorm2d(128) if conv_batch_norm else Identity(),
            conv_activation_class(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            BatchNorm2d(256) if conv_batch_norm else Identity(),
            conv_activation_class(),
            AdaptiveMaxPool2d(output_size=(6, 6)),
            
            Flatten(),
            BatchNorm1d(self.encoding_dim) if linear_batch_norm else Identity(),
            Linear(in_features=9216, out_features=self.encoding_dim),
            final_activation_class() if final_activation_class else Identity()
        )

        self.weights_init()

    def weights_init(self):
        def weights_init_(m: Module):
            if isinstance(m, Conv2d):
                init.normal_(m.weight, self.conv_weight_mean, self.conv_weight_std)
                if m.bias is not None:
                    init.normal_(m.bias, self.conv_bias_mean, self.conv_bias_std)
            elif isinstance(m, Linear):
                init.normal_(m.weight, self.linear_weight_mean, self.linear_weight_std)
                if m.bias is not None:
                    init.normal_(m.bias, self.linear_bias_mean, self.linear_bias_std)

        self.network.apply(weights_init_)

    def forward(self, img):
        return self.network(img)
    