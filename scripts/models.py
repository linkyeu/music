from common import *


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()       
        model = models.resnet18(True)
        self.bone = list(model.children())[:-2]
        self.bottle = self.bone[-1][-1] # Bottleneck
        self.n_ft = list(self.bottle.children())[-1].num_features
        self.head = nn.Sequential( 
            *self.bone,
            AdaptiveConcatPool2d(sz=1),
            Flatten(),
            nn.BatchNorm1d(num_features=self.n_ft*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=self.n_ft*2, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=102, bias=True),
        )                    
    def forward(self, x):
        output = self.head(x)        
        return output