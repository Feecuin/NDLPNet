## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from PIL import Image
from einops import rearrange
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # ##print(mu.size())
        # ##print(sigma.size())
        # ##print(x.shape)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)       

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,
        dim =128        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, 48)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2*2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2*2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2*2*2), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2*2*2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2*2*2), int(dim*2*2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2*2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2*2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2*2), int(dim*2*1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2*1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        

            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


        self.prior = PReNet_for_RLP()
        self.positionencode = rain_map_positionencoding(num_pos_feats_x=32, num_pos_feats_y=32, num_pos_feats_z=16)
        self.pos_drop = nn.Dropout(p=0.)
        self.ca = ChannelAttentionModule(in_channels = 128)
        
    def forward(self, inp_img):


        rain_map = self.prior(inp_img)
        # rain_map_to_visualize = rain_map[0]
        # os.makedirs("real-rain", exist_ok=True)
        # for channel in range(rain_map_to_visualize.shape[0]):
        #     # 获取当前通道的特征图
        #     feature_map = rain_map_to_visualize[channel]

        #     # 将张量转换为 numpy 数组
        #     feature_map_np = feature_map.detach().cpu().numpy()

        #     # 归一化到 [0, 1] 范围
        #     feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min())

        #     # 使用 matplotlib 绘制特征图
        #     plt.imshow(feature_map_np, cmap='magma')
        #     plt.axis('off')
        #     # plt.colorbar()
        #     # plt.title(f'Channel {channel}')

        #     # 保存图像
        #     plt.savefig(f"real-rain/channel_{channel}.png",bbox_inches='tight', pad_inches=0)
        #     plt.close()







        x = self.patch_embed(inp_img)


        Wh, Ww = x.size(2), x.size(3)
        depth_pool =  F.interpolate(rain_map, size=(Wh, Ww), mode='bicubic')   
        # ##print("depth_pool",depth_pool.shape)

        positioncoding = self.positionencode(x,depth_pool)



        # ##print("positioncoding",positioncoding.shape)
        x = torch.cat((x,positioncoding),dim=1)

        
        x = self.pos_drop(x)
        # print("xdrop",x.shape)

        x = self.ca(x)
        # visualize_with_pca(x)
        # visualize_strongest_channels(x)
        # save_attention_map(x,batch_idx=0, channel_idx=0)

        # inp_enlevel1 = self.patch_embed(inp_img)
        # ##print(inp_enc_level1.shape)
        # out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(x)
        # print("out_enc_level",out_enc_level1.shape)

        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        # print("inp_enc_level2",inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # print("out_enc_level2",out_enc_level2.shape)


        inp_enc_level3 = self.down2_3(out_enc_level2)
        # print("inp_enc_level3",inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        #print("out_enc_level3",out_enc_level3.shape)


        inp_enc_level4 = self.down3_4(out_enc_level3)
        # print("inp_enc_level4",inp_enc_level4.shape)        
        latent = self.latent(inp_enc_level4) 
        # print("latent",latent.shape)


        inp_dec_level3 = self.up4_3(latent)
        # print("inp_dec_level3",inp_dec_level3.shape)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        #print("inp_dec_level3cat",inp_dec_level3.shape)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        #print("inp_dec_level3",inp_dec_level3.shape)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        #print("out_dec_level3",out_dec_level3.shape)


        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        #print("inp_dec_level2",inp_dec_level2.shape)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        #print("inp_dec_level2",inp_dec_level2.shape)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        #print("inp_dec_level2",inp_dec_level2.shape)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        ##print("out_dec_level2",out_dec_level2.shape)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        ##print("inp_dec_level1",inp_dec_level1.shape)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        ##print("inp_dec_level1",inp_dec_level1.shape)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        ##print("out_dec_level11",out_dec_level1.shape)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        ##print("out_dec_level1",out_dec_level1.shape)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1




class PReNet_for_RLP(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_for_RLP, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        rlp = Variable(torch.ones(batch_size, 1, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            rlp = rlp.cuda()

        # rlp_list = []
        for i in range(self.iteration):
            x = torch.cat((input, rlp), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            rlp = self.conv(x)

            # rlp_list.append(rlp)

        return rlp
    




class rain_map_positionencoding(nn.Module):
    def __init__(self, num_pos_feats_x=128, num_pos_feats_y=128, num_pos_feats_z=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats_x = num_pos_feats_x
        self.num_pos_feats_y = num_pos_feats_y
        self.num_pos_feats_z = num_pos_feats_z
        self.num_pos_feats = max(num_pos_feats_x, num_pos_feats_y, num_pos_feats_z)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, depth):
        b, c, h, w = x.size()
        b_d, c_d, h_d, w_d = depth.size()
        assert b == b_d and c_d == 1 and h == h_d and w == w_d
        
        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            y_embed = torch.arange(h, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(b, 1, w)
            x_embed = torch.arange(w, dtype=torch.float32, device=x.device).repeat(b, h, 1)
        z_embed = depth.squeeze().to(dtype=torch.float32, device=x.device)

        if self.normalize:
            eps = 1e-6
            if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
                y_embed = y_embed / (y_embed.max() + eps) * self.scale
                x_embed = x_embed / (x_embed.max() + eps) * self.scale
            z_embed_max, _ = z_embed.reshape(b, -1).max(1)
            z_embed = z_embed / (z_embed_max[:, None, None] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos_x = x_embed[:, :, :, None] / dim_t[:self.num_pos_feats_x]
            pos_y = y_embed[:, :, :, None] / dim_t[:self.num_pos_feats_y]
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos_z = z_embed[:, :, :, None] / dim_t[:self.num_pos_feats_z]
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3)

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos = torch.cat((pos_x, pos_y, pos_z), dim=3).permute(0, 3, 1, 2)
        else:
            pos = pos_z.permute(0, 3, 1, 2)
        return pos


# class ChannelAttentionModule(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         self.reduction_ratio = reduction_ratio
#         self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         squeeze = F.adaptive_avg_pool2d(x, output_size=1)
#         excitation = self.fc1(squeeze)
#         excitation = self.relu(excitation)
#         excitation = self.fc2(excitation)
#         attention_weights = self.sigmoid(excitation)
#         out = x * attention_weights.unsqueeze(-1).unsqueeze(-1)
#         return out
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用 adaptive_avg_pool2d 并保持原有维度
        squeeze = F.adaptive_avg_pool2d(x, output_size=(1, 1))

        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        attention_weights = self.sigmoid(excitation)

        # 不需要额外重塑，因为squeeze已经是[batch_size, channels, 1, 1]
        out = x * attention_weights

        return out
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         self.reduction_ratio = reduction_ratio
#         self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         squeeze = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, channels)
#         excitation = self.fc1(squeeze)
#         excitation = self.relu(excitation)
#         excitation = self.fc2(excitation)
#         attention_weights = self.sigmoid(excitation).view(batch_size, channels, 1, 1)
        
#         out = x * attention_weights.expand_as(x)
        
#         return out
