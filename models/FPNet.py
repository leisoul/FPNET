import torch
import torch.nn as nn
import torch.nn.functional as F
if not __package__:
    from NAFNet_archs.NAFNet_arch import  NAFNet_backbone
else:
    from models.NAFNet_archs.NAFNet_arch import  NAFNet_backbone

    
class CEFF(nn.Module):
    # chnnel enhance feed foreword
    def __init__(self, in_channel,):
        super(CEFF, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        #liinear
        self.conv = nn.Conv2d(in_channels=in_channel , out_channels=in_channel , kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channels=in_channel , out_channels=in_channel*2 , kernel_size=1)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        y = self.norm(y)
        y = self.conv2(y)

        return y

class DK(nn.Module): 
    def __init__(self, width):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(width)
        self.down = nn.MaxPool2d(2, 2)


        self.dilated_conv = nn.Conv2d(in_channels=width,out_channels=width,kernel_size=3,
                                      padding=2, dilation=2,bias=True)
        
        self.bn2 = nn.BatchNorm2d(width)
        self.Tanh = nn.Tanh()

        self.ceff = CEFF(width)

        # self.pool = nn.MaxPool2d(64)
        self.down_C = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, padding=0, groups=1, bias=True)


    def forward(self, inp, wieght):
        B, C, H, W = inp.shape
        wieght = 1 - wieght

        x = self.bn1(inp)
        x = self.down(x)
        x = self.dilated_conv(x)
        y = self.bn2(x)
        y = self.Tanh(y)

        x = x * y

        
        x = self.ceff(x)

        # x = self.pool(x)
        x = self.down_C(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        
        x = x * wieght
        x = x * inp
        return x

class FeaturePick(nn.Module):
    def __init__(self, in_channel, hidden, width, FGM_nums=4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=hidden, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.down = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=2, padding=0, stride=2, groups=1, bias=True)
        
        self.bn = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU()

        self.FGMs = nn.ModuleList()
        chan = hidden
        for num in range(FGM_nums):
            self.FGMs.append(
                nn.Sequential(
                    *[FGM(chan)]
                )
            )
            chan = chan * 2

        self.down_C = nn.Conv2d(in_channels=chan, out_channels=width, kernel_size=1, padding=0, groups=1, bias=True)
        # self.pool = nn.MaxPool2d(int(128 / 2**FGM_nums))

    def forward(self, inp):
        B, C, H, W = inp.shape

        x = self.conv(inp)
        x = self.bn(x)
        x = self.down(x)
        x = self.relu(x)

        for FGM in self.FGMs:
            x = FGM(x)

        x = self.down_C(x)
        # x = self.pool(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        x = torch.tanh(x)
        
        
        return x

class FGM(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.dep_conv = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1, stride=1, groups=hidden, bias=True)


        self.ceff = CEFF(hidden)

    def forward(self, inp):
        B, C, H, W = inp.shape

        x = self.bn2(inp)
        x = self.dep_conv(x)

        inp = self.bn1(inp)
        inp = self.conv(inp)


        x = x * inp

 
        
        x = self.ceff(x)

        
        return x

class FPNet(nn.Module):

    def __init__(self,inp_size=256, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], hidden=4, FGM_nums=4, backbone=None):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.feature_pick = FeaturePick(img_channel,  hidden, width, FGM_nums)
        self.DK = DK( width)
        if backbone == 'NAFNet':
            self.backbone = NAFNet_backbone( img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)



    def forward(self, inp):
        B, C, H, W = inp.shape

        x = self.intro(inp)

        weight = self.feature_pick(inp)
        y = self.DK(x, weight)

        x = x * weight

        x =  self.backbone(x)

        x =  x + y
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

