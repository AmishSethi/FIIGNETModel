import itertools
import torch
from torch import nn
from torch.nn import functional as F
from typing import List
import numpy as np


import distributed as dist_fn


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)


        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out



class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        blocks = []
        for i in range(n_res_block):
            blocks.append(ResBlock(in_channel, n_res_channel))

        if stride == 4:
            blocks.extend( 
                [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1)
                ]
            )

        elif stride == 2:
            blocks.extend(
                [
                    nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 2, channel, 3, padding=1)
                ]
            )
                

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        for i in range(n_res_block):
            blocks.append(ResBlock(out_channel, n_res_channel))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(256*256, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 256 * 256)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self,  input_dim, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, self.input_dim, self.input_dim)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDecoder, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class CustomEncoder(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 hidden_dims: List = [32, 64, 128],
                 anneal_steps: int = 200,
                 ):
        super().__init__()
        self.anneal_steps = anneal_steps

        modules = []
        
        hidden_dims = [32, 64, 128, out_channels]
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        



        self.encoder = nn.Sequential(*modules)

    def forward(self, input):
        result = self.encoder(input)
        return result

class CustomDecoder(nn.Module):
    def __init__(self,
                 in_channel, out_channel):
        super().__init__()
        modules = []

        hidden_dims = [in_channel//8, in_channel//4, in_channel//2, in_channel]




        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(int(hidden_dims[i]),
                                       int(hidden_dims[i + 1]),
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channel,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def forward(self, input):
        result = self.decoder(input)
        #result = self.final_layer(result)
        return result


class FIIGNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=128,
        n_res_block=3,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()


        self.enc_b = Encoder(in_channel, out_channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(out_channel, out_channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(out_channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, out_channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + out_channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
                    embed_dim + embed_dim,
                    in_channel,
                    out_channel,
                    n_res_block,
                    n_res_channel,
                    stride=4,
        )
 

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        # recons_features = self.extract_features(dec)
        # input_features = self.extract_features(input)

        return dec, diff #, recons_features, input_features

    def encode(self, input):
        enc_b = self.enc_b(input)
        # print("enc_b shape", enc_b.shape)
        enc_t = self.enc_t(enc_b)
        # print("enc_t shape", enc_t.shape)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        # print("quant_t shape", quant_t.shape)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        # print("dec_t shape", dec_t.shape)
        enc_b = torch.cat([dec_t, enc_b], 1)
        # print("enc_b shape after cat", enc_b.shape)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        # print("upsample_t shape", upsample_t.shape)
        quant = torch.cat([upsample_t, quant_b], 1)
        # print("quant shape", quant.shape)
        dec = self.dec(quant)
        # print("dec shape", dec.shape)


        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
    
    # def extract_features(self,
    #                      input):
    #     features = []

    #     enc_b = self.enc_b(input)
    #     enc_t = self.enc_t(enc_b)

    #     features.append(enc_t)

    #     quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
    #     quant_t, diff_t, id_t = self.quantize_t(quant_t)
    #     quant_t = quant_t.permute(0, 3, 1, 2)
    #     diff_t = diff_t.unsqueeze(0)

    #     features.append(quant_t)

    #     dec_t = self.dec_t(quant_t)
    #     enc_b = torch.cat([dec_t, enc_b], 1)

    #     features.append(enc_b)

    #     quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
    #     quant_b, diff_b, id_b = self.quantize_b(quant_b)
    #     quant_b = quant_b.permute(0, 3, 1, 2)
    #     diff_b = diff_b.unsqueeze(0)

    #     features.append(quant_b)

    #     upsample_t = self.upsample_t(quant_t)

    #     features.append(upsample_t)

    #     quant = torch.cat([upsample_t, quant_b], 1)

    #     features.append(quant)
        

    #     return features

