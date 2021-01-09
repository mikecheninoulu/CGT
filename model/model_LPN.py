from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PoseFeature(nn.Module):
    def __init__(self, num_points = 6890):
        super(PoseFeature, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(1024)

    def forward(self, x):

        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        return x


class SPAdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(SPAdaIN,self).__init__()
        self.conv_weight = nn.Conv1d(input_nc, planes, 1)
        self.conv_bias = nn.Conv1d(input_nc, planes, 1)
        self.norm = norm(planes)

    def forward(self,x,addition):

        x = self.norm(x)
        weight = self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out

class SPAdaINResBlock(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(SPAdaINResBlock,self).__init__()
        self.spadain1 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain2 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain_res = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv_res=nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self,x,addition):

        out = self.spadain1(x,addition)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spadain2(out,addition)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        residual = self.spadain_res(residual,addition)
        residual = self.relu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return  out

class Self_NonLocalBlock(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(Self_NonLocalBlock,self).__init__()

        self.query_conv = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.key_conv = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.value_conv = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, id_f):
        m_batchsize, C, p_len = id_f.size()
        #[8, 1024, 6890]
        proj_query = self.query_conv(id_f).permute(0, 2, 1)
        proj_key = self.key_conv(id_f)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(id_f)
        value_attention = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = id_f + self.gamma*value_attention  # connection

        return  out

class Decoder(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Decoder, self).__init__()

        self.nonlocal_block = Self_NonLocalBlock(input_nc=3 ,planes=self.bottleneck_size-3)

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.spadain_block1 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size)
        self.spadain_block2 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.spadain_block3 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//4)

        self.norm1 = torch.nn.InstanceNorm1d(self.bottleneck_size)
        self.norm2 = torch.nn.InstanceNorm1d(self.bottleneck_size//2)
        self.norm3 = torch.nn.InstanceNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()


    def forward(self, x1, x1_f):
        y = self.nonlocal_block(x1_f)
        y = torch.cat((y, x1), 1)
        x = self.conv1(y)
        x = self.spadain_block1(x,x1)
        x = self.conv2(x)
        x = self.spadain_block2(x,x1)
        x = self.conv3(x)
        x = self.spadain_block3(x,x1)
        x = 2*self.th(self.conv4(x))

        return x

class NPT(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024):
        super(NPT, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = PoseFeature(num_points = num_points)
        self.decoder = Decoder(bottleneck_size = self.bottleneck_size+3)

    def forward(self, x1):
        #[8, 3, 6890]
        x1_f = self.encoder(x1)
        #[8, 1024, 6890]
        out =self.decoder(x1, x1_f)

        return out.transpose(2,1)
