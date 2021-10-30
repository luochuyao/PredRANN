import torch
import torch.nn as nn

class TimeDistribution(nn.Module):
    def __init__(self,model):
        super(TimeDistribution, self).__init__()
        self.model = model

    def forward(self, input):
        t_length = input.shape[1]
        outputs = []
        for t in range(t_length):
            outputs.append(self.model(input[:,t]))
        outputs = torch.stack(outputs,1)
        return outputs