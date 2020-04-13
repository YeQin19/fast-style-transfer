from collections import namedtuple
import torch


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class FastNSTLoss(torch.nn.Module):
    def __init__(self, vgg_model, lossOutput):
        super(FastNSTLoss, self).__init__()
        self.vgg_layers = vgg_model.features
        self.lossOutput = lossOutput
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.lossOutput(**output)