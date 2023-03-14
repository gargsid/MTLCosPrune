from typing import List, cast

import torch
import torch.nn as nn
from nyuv2_data.pixel2pixel import ASPPHeadNode, SmallASPPHeadNode

from .gate_layer import GateLayer

class Adapter(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        conv2d_p = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        conv2d_q = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        
        layers: List[nn.Module] = [conv2d_p, nn.ReLU(inplace=True), conv2d_q]
        self.adapter = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

class FineConv2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, first_layer: bool = False):
        super().__init__() 
        if not first_layer:
            self.adapter_up = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.adapter_down = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'adapter_up'):
            return self.adapter_down(self.conv2d(self.adapter_up(x)))
        else:
            return self.adapter_down(self.conv2d(x))
        
class FinalConv2d(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.final_adapter = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_adapter(x)
    
class VGG(nn.Module):
    def __init__(
        self, out_channels: list = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], 
        num_classes: int = 10, batch_norm: bool = False, init_weights: bool = True, dropout: float = 0.5, adapter: int = 0
    ) -> None:
        super().__init__()
        
        layers: List[nn.Module] = []
        in_channels = 3
        first_layer = True
        for vidx, v in enumerate(out_channels):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                prev_layer_out = out_channels[vidx-1]

                # added gate layer for taylor pruning
                layers += [GateLayer(prev_layer_out, prev_layer_out, [1, -1, 1, 1])]

            else:
                v = cast(int, v)
                lst = []

                if adapter == 2:
                    lst.append(FineConv2d(in_channels, v, first_layer))
                else:
                    lst.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))

                if batch_norm:
                    lst.append(nn.BatchNorm2d(v))
                
                if adapter == 1:
                    lst.append(Adapter(v))
                else:
                    lst.append(nn.ReLU(inplace=True))

                    if vidx < len(out_channels) and out_channels[vidx+1] != 'M':
                        # added gate layer for taylor pruning
                        lst.append(GateLayer(v, v, [1, -1, 1, 1]))
                
                layers += lst
                in_channels = v
                first_layer = False
                    
        if adapter == 2: # for heads
            layers.append(FinalConv2d(in_channels))
                
        self.backbone = nn.Sequential(*layers)
        self.head = SmallASPPHeadNode(in_channels, num_classes)
        
        if init_weights:
            for n, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'adapter' not in n:
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    else:
                        # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                        # nn.init.orthogonal_(m.weight)
                        nn.init.dirac_(m.weight) # identity matrix
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
    
class MTLNet(nn.Module):
    def __init__(self, tasks, cls_dict, s_config=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], backbone='vgg16', init_weights: bool = True): # cls_dict: a dictionary
        super(MTLNet, self).__init__()
        self.tasks = tasks
        if backbone == 'vgg16':
            self.backbone = VGG(self.gen_out(s_config)).backbone
        elif backbone == 'vgg16-ada1' or backbone == 'vgg16-ada2':
            self.backbone = VGG(self.gen_out(s_config), adapter=int(backbone[-1])).backbone
        else:
            print('Unsupported Backbone Model!', flush=True)
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = SmallASPPHeadNode(s_config[-1], cls_dict[task])
            
    def forward(self, x):
        out = {task: self.heads[task](self.backbone(x)) for task in self.tasks}
        # out = [self.heads[task](self.backbone(x)) for task in self.tasks]
        return out
    
    def gen_out(self, s_config, pos=[2,5,9,13,17]):
        out = list(s_config)
        for p in pos:
            out.insert(p,"M")
        return out
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def freeze_backbone_except_adapter(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
            