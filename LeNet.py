import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, number_of_classes, in_features):
        super(LeNet, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            nn.Sigmoid(),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(in_features, number_of_classes)
        )
        self.apply(init_weights)
        

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.final_layer(out)
        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.bias.data.uniform_(-0.5, 0.5)
        m.weight.data.uniform_(-0.5, 0.5)