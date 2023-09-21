import torch
from torch import nn
import torch.nn.functional as F


class SpatAttn(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy = torch.bmm(proj_query, proj_key)  # BxHWxHW, attention maps
        attention = self.softmax(energy)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW
        out = out.view(m_batchsize, C, height, width)  # BxCxHxW

        out = self.gamma * out + x
        return out


class SpatAttn_(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn_, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.Sequential(nn.ReLU(),
                                nn.BatchNorm2d(in_dim))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy = torch.bmm(proj_query, proj_key)  # BxHWxHW, attention maps
        attention = self.softmax(energy)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW
        out = out.view(m_batchsize, C, height, width)  # BxCxHxW

        out = self.gamma * out  # + x
        return self.bn(out)

class SARes(nn.Module):
    def __init__(self, in_dim, ratio=8, resin=False):
        super(SARes, self).__init__()

        if resin:
            self.sa1 = SpatAttn(in_dim, ratio)
            self.sa2 = SpatAttn(in_dim, ratio)
        else:
            self.sa1 = SpatAttn_(in_dim, ratio)
            self.sa2 = SpatAttn_(in_dim, ratio)

    def forward(self, x):
        identity = x
        x = self.sa1(x)
        x = self.sa2(x)

        return F.relu(x + identity)


class SPC32(nn.Module):
    def __init__(self, msize=24, outplane=49, kernel_size=[7, 1, 1], stride=[1, 1, 1], padding=[3, 0, 0], spa_size=9,
                 bias=True):
        super(SPC32, self).__init__()

        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask0
        self.bn1 = nn.BatchNorm2d(outplane)

        self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask2
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x  # NCHW
        n, c, h, w = identity.size()

        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
        mask0 = torch.softmax(mask0.view(n, -1, h * w), -1)
        mask0 = mask0.view(n, -1, h, w)
        _, d, _, _ = mask0.size()

        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)  # NCD

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0)  # NCDHW

        out = F.leaky_relu(out)
        out = out.sum(2)

        out = out  # + identity

        out0 = self.bn1(out.view(n, -1, h, w))

        mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
        mask2 = torch.softmax(mask2.view(n, -1, h * w), -1)
        mask2 = mask2.view(n, -1, h, w)

        fk = torch.einsum('ndhw,nchw->ncd', mask2, x)  # NCD

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2)  # NCDHW

        out = F.leaky_relu(out)
        out = out.sum(2)

        out = out + identity

        out = self.bn2(out.view(n, -1, h, w))

        return out  # NCHW


class SSNet_AEAE(nn.Module):
    '''
        GitHub repo: https://github.com/zilongzhong/SSTN
    '''
    def __init__(self, num_classes=16, num_channels=None, msize=16, inter_size=49):
        super(SSNet_AEAE, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        #         self.layer1 = SPCModuleIN_(1, 1, inter_size=inter_size)
        #         self.bn1 = nn.BatchNorm2d(inter_size)
        self.layer1 = nn.Sequential(nn.Conv2d(self.num_channels, inter_size, 1),
                                    # nn.LeakyReLU(),
                                    nn.BatchNorm2d(inter_size), )
        # nn.LeakyReLU())
        #         self.layer1 = SPC1d(stride=[2,1], padding=[0,0])

        self.layer2 = SARes(inter_size, ratio=8)  # ResSPA(inter_size, inter_size)
        self.layer3 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])

        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)

        self.layer5 = SARes(msize, ratio=8)  # ResSPA(msize, msize)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize, 1, 1], padding=[0, 0, 0])

        self.fc = nn.Linear(msize, self.num_classes)

    def forward(self, x, meta_data=None):
        n, c, h, w = x.size()

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer31(x)

        #         x = x.contiguous()
        #         x = x.reshape(n,-1,h,w)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        # x = self.fc(x.squeeze())
        x = self.fc(torch.flatten(x, start_dim=1))

        return x