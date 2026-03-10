import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from models.model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from models.HGNN import HyperComputeModule, HyPConv
from models.build import MODELS

class Harmonic3DPE(nn.Module):
    def __init__(self, num_bands: int = 6):
        super().__init__()
        self.num_bands = num_bands
        self.freq_bands = nn.Parameter(torch.pow(2, torch.arange(self.num_bands, dtype=torch.float32)))

    @property
    def out_dim(self):
        return 3 + 2 * self.num_bands * 3

    def forward(self, xyz, channels_last: bool = True):
        if not channels_last:
            xyz = xyz.transpose(1, 2).contiguous()
        B, N, _ = xyz.shape
        fb = self.freq_bands.to(xyz.device)
        ang = xyz.unsqueeze(-1) * fb
        pe = torch.cat([xyz, torch.sin(ang).reshape(B, N, -1), torch.cos(ang).reshape(B, N, -1)], dim=-1)
        return pe

class PyramidSAK(nn.Module):
    def __init__(self, base_npoint=512, base_k=16, ms_knn_list=(8, 16, 24), if_bn=False):
        super().__init__()
        self.sa = PointNet_SA_Module_KNN(base_npoint, base_k, 3, [64, 128], group_all=False, if_bn=if_bn, if_idx=True)
        self.ms_mlps = nn.ModuleList([MLP_CONV(in_channel=3, layer_dims=[64, 128]) for _ in ms_knn_list])
        self.ms_knn_list = ms_knn_list

    def forward(self, l0_xyz, l0_points):
        keypoints, base_feat, _ = self.sa(l0_xyz, l0_points)

        B, _, N = l0_xyz.shape
        _, _, S = keypoints.shape
        with torch.no_grad():
            dist = torch.cdist(keypoints.transpose(2, 1).contiguous(),
                               l0_xyz.transpose(2, 1).contiguous())

        ms_feats = [base_feat]
        for k, mlp in zip(self.ms_knn_list, self.ms_mlps):
            k_eff = min(k, N)
            idx = dist.topk(k_eff, largest=False, dim=-1).indices
            nbr = torch.gather(
                l0_xyz.transpose(2, 1).unsqueeze(1).expand(B, S, N, 3),
                2, idx.unsqueeze(-1).expand(B, S, k_eff, 3)
            )
            center = keypoints.transpose(2, 1).unsqueeze(2)
            rel = (nbr - center)
            rel_feat = rel.reshape(B, S * k_eff, 3).transpose(2, 1).contiguous()
            rel_feat = mlp(rel_feat)
            rel_feat = rel_feat.reshape(B, 128, S, k_eff).max(dim=-1)[0]
            ms_feats.append(rel_feat)

        return keypoints, torch.cat(ms_feats, dim=1)

class HyperRS(nn.Module):
    def __init__(self, c, L=6, thr_start=0.20, thr_end=0.16):
        super().__init__()
        self.layers = nn.ModuleList([HyperComputeModule(c, c) for _ in range(L)])
        self.L = L
        self.thr_start = thr_start
        self.thr_end = thr_end

    def forward(self, x):
        B, N, C = x.shape
        thr_list = torch.linspace(self.thr_start, self.thr_end, steps=self.L, device=x.device, dtype=x.dtype)
        y = None
        for i, layer in enumerate(self.layers):
            layer.threshold = float(thr_list[i].item())
            y = layer(x)
            x = y.transpose(2, 1).contiguous()
        return y

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        _, NK, _ = y.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

class CrossFormer(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, out_dim))

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x

class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)

    def forward(self, feat_x, feat_y):
        feat = self.corssformer_1(feat_x, feat_y)
        feat = self.corssformer_2(feat, feat)
        return feat

class AHGNN(nn.Module):
    def __init__(self, c1, c2, topk=24):
        super().__init__()
        self.topk = topk
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()

    @staticmethod
    def uniform_subsample(coords, M):
        B, N, _ = coords.shape
        if M >= N:
            return coords
        idx = torch.linspace(0, N - 1, steps=M, device=coords.device).long()
        return coords[:, idx, :]

    def build_anchor_incidence(self, coords, anchors):
        with torch.no_grad():
            dist = torch.cdist(coords, anchors)
            k = min(self.topk, anchors.shape[1])
            topk_idx = dist.topk(k, largest=False, dim=-1).indices
            H = torch.zeros_like(dist)
            H.scatter_(-1, topk_idx, 1.0)
        return H

    def forward(self, x, coords, anchors):
        H = self.build_anchor_incidence(coords, anchors)
        x = self.hgconv(x, H) + x
        x = x.transpose(1, 2).contiguous()
        x = self.act(self.bn(x))
        return x

class HyperEncoder(nn.Module):
    def __init__(self, out_dim=512, num_bands_pe: int = 6, ms_knn_list=(8, 16, 24), anchor_M1: int = 128, anchor_M2: int = 192):
        super(HyperEncoder, self).__init__()

        self.sa_ms = PyramidSAK(base_npoint=512, base_k=16, ms_knn_list=ms_knn_list, if_bn=False)
        self.ms_out_channels = 128 * (1 + len(ms_knn_list))

        self.pe = Harmonic3DPE(num_bands=num_bands_pe)
        self.pe_dim = self.pe.out_dim

        self.pe_proj = MLP_CONV(in_channel=self.ms_out_channels + self.pe_dim, layer_dims=[self.ms_out_channels])
        self.transformer_1 = Transformer(self.ms_out_channels, dim=64)

        self.expanding = MLP_CONV(in_channel=self.ms_out_channels, layer_dims=[256, out_dim])
        self.keyfeat_down = MLP_CONV(in_channel=self.ms_out_channels, layer_dims=[128])

        self.hyper_stack = HyperRS(1024, L=6, thr_start=0.20, thr_end=0.16)

        self.anchor_hyper_1 = AHGNN(1024, 1024, topk=24)
        self.anchor_hyper_2 = AHGNN(1024, 1024, topk=32)
        self.anchor_M1 = anchor_M1
        self.anchor_M2 = anchor_M2

        self.mlp = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 12)
        )

    def forward(self, point_cloud):
        b = point_cloud.shape[0]

        l0_xyz = point_cloud.transpose(2, 1).contiguous()
        l0_points = l0_xyz

        keypoints, ms_feats = self.sa_ms(l0_xyz, l0_points)
        kp_xyz = keypoints.transpose(2, 1).contiguous()

        kp_pe = self.pe(kp_xyz, channels_last=True)

        x = torch.cat([ms_feats.transpose(2, 1).contiguous(), kp_pe], dim=-1)
        x = x.transpose(2, 1).contiguous()

        keyfeatures_ms = self.pe_proj(x)
        keyfeatures_ms = self.transformer_1(keyfeatures_ms, keypoints)

        feat = self.expanding(keyfeatures_ms).transpose(2, 1).contiguous()
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(1), 1)], dim=-1)

        feat = self.hyper_stack(feat)
        feat = feat.transpose(2, 1).contiguous()

        anchors1 = AHGNN.uniform_subsample(kp_xyz, self.anchor_M1)
        feat = self.anchor_hyper_1(feat, kp_xyz, anchors1)
        feat = feat.transpose(2, 1).contiguous()

        anchors2 = AHGNN.uniform_subsample(kp_xyz, self.anchor_M2)
        feat = self.anchor_hyper_2(feat, kp_xyz, anchors2)
        feat = feat.transpose(2, 1).contiguous()

        ret = self.mlp(feat)
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]

        symmetry_points = torch.matmul(kp_xyz.unsqueeze(2), R).view(b, 512, 3) + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()

        coarse = torch.cat([symmetry_points, keypoints], dim=-1)
        keyfeatures_128 = self.keyfeat_down(keyfeatures_ms)

        return coarse, symmetry_points, keyfeatures_128


class CFGTransformer(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2, num_bands_pe: int = 6):
        super(CFGTransformer, self).__init__()
        self.up_factor = up_factor

        self.pe = Harmonic3DPE(num_bands=num_bands_pe)
        self.pe_dim = self.pe.out_dim

        self.mlp_1 = MLP_CONV(in_channel=3 + self.pe_dim, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)

        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)

        self.mlp_fusion = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.2), nn.Linear(512, 512))
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape

        pe = self.pe(coarse.transpose(2, 1).contiguous(), channels_last=True)
        inp = torch.cat([coarse.transpose(2, 1).contiguous(), pe], dim=-1).transpose(2, 1).contiguous()
        feat = self.mlp_1(inp)

        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)

        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        feat_p = self.fusion_1(feat, partial_feat)
        feat_s = self.fusion_2(feat, symmetry_feat)

        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)
        feat = self.fusion_3(feat, feat)

        offset = self.fc(feat).view(b, -1, 3)
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        return pcd_up


class local_encoder(nn.Module):
    def __init__(self, out_channel=128, num_bands_pe: int = 6):
        super(local_encoder, self).__init__()
        self.pe = Harmonic3DPE(num_bands=num_bands_pe)
        self.pe_dim = self.pe.out_dim
        self.mlp_1 = MLP_CONV(in_channel=3 + self.pe_dim, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self, input):
        pe = self.pe(input.transpose(2, 1).contiguous(), channels_last=True)
        x = torch.cat([input.transpose(2, 1).contiguous(), pe], dim=-1).transpose(2, 1).contiguous()
        feat = self.mlp_1(x)
        feat = torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, input)
        return feat

@MODELS.register_module(force=True)
class Hyper_PCN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]

        self.encoder = HyperEncoder(out_dim=512)

        self.local_encoder = local_encoder(out_channel=128)
        self.cfg_transformer_1 = CFGTransformer(gf_dim=512, up_factor=self.up_factors[0])
        self.cfg_transformer_2 = CFGTransformer(gf_dim=512, up_factor=self.up_factors[1])

        self.include_input = config.include_input
        self.loss_func = ChamferDistanceL1()

        self.sym_anchor_hyper = AHGNN(128, 128, topk=8)

    def get_loss(self, rets, gt):
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total, loss_list[0], loss_list[-1], loss_list[0], loss_list[-1]

    def forward(self, point_cloud):
        coarse, symmetry_points, keyfeatures_128 = self.encoder(point_cloud)

        keypoints_coords = coarse[:, :, 512:]
        nodes_coords = keypoints_coords.transpose(2, 1).contiguous()
        sym_anchors = symmetry_points.transpose(2, 1).contiguous()
        partial_nodes = keyfeatures_128.transpose(2, 1).contiguous()

        keyfeatures_128_refined = self.sym_anchor_hyper(partial_nodes, nodes_coords, sym_anchors)

        feat_symmetry = self.local_encoder(symmetry_points)
        feat_partial = keyfeatures_128_refined

        fine1 = self.cfg_transformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.cfg_transformer_2(fine1.transpose(2, 1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input:
            fine2 = torch.cat([fine2, point_cloud], dim=1).contiguous()

        rets = [coarse.transpose(2, 1).contiguous(), fine1, fine2]
        self.pred_dense_point = rets[-1]
        return rets