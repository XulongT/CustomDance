"""Shared inpainting network adapted from FlowerDance."""

import math
import torch
from einops import rearrange
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        frequencies = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -frequencies)
        phase = x[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)



class DenseFiLM(nn.Module):
    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(nn.Mish(), nn.Linear(embed_channels, embed_channels * 2))

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class IntraModalMamba(nn.Module):
    def __init__(
        self,
        d_model,
        dropout,
        layer_norm_eps,
    ):
        super().__init__()
        self.mamba_forward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_backward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.mamba_forward(x)
        x2 = self.mamba_backward(x.flip(1)).flip(1)
        x = self.dropout(x1 + x2)
        return x


class DanceFiLMLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
    ):
        super().__init__()

        self.intra_mamba = IntraModalMamba(d_model, dropout, layer_norm_eps)

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)

    # x, cond, t
    def forward(self, x, cond, genre, t):
        # intra-modal -> film -> residual
        x_1 = self.intra_mamba(x)
        x = x + featurewise_affine(x_1, self.film1(t))

        # cross-modal -> film -> residual
        x = x + featurewise_affine(cond, self.film2(t))
        return x


class GenreGate(nn.Module):
    def __init__(self, num_genres: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_genres, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        self.delta = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.alpha = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        nn.init.constant_(self.alpha[-1].bias, -4.0)

    def forward(self, x, genre):
        # x: [B,T,C], genre: [B]
        g = self.proj_g(self.emb(genre))[:, None, :]  # [B,1,C]
        z = torch.cat([x, g.expand(-1, x.size(1), -1)], -1)  # [B,T,2C]
        delta = self.delta(z)  # [B,T,C]
        alpha = torch.sigmoid(self.alpha(z))  # [B,T,C]
        return x + alpha * delta


class MusicFiLMLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
    ):
        super().__init__()

        self.intra_mamba = IntraModalMamba(d_model, dropout, layer_norm_eps)
        self.genre_gate = GenreGate(16, d_model)
        self.film = DenseFiLM(d_model)

    def forward(self, x, genre):
        x = self.genre_gate(x, genre)
        x = self.intra_mamba(x)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats,
        seq_len=150,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.2,
        cond_feature_dim=4800,
        activation=F.gelu,
        use_rotary=True,
        **kwargs,
    ):
        super().__init__()

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.intv_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

        # input and condition projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.final_layer = nn.Linear(latent_dim, nfeats)

        # Dance decoder
        self.dance_num_layers = 8
        self.dancelayerstack = nn.ModuleList([])
        for _ in range(self.dance_num_layers):
            self.dancelayerstack.append(
                DanceFiLMLayer(
                    latent_dim, num_heads, ff_size, dropout, activation, batch_first=True
                )
            )

        # Music encoder
        self.music_num_layers = 4
        self.musiclayerstack = nn.ModuleList([])
        for _ in range(self.music_num_layers):
            self.musiclayerstack.append(
                MusicFiLMLayer(
                    latent_dim, num_heads, ff_size, dropout, activation, batch_first=True
                )
            )

        self.null_cond_embed = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.guidance_weight = 4

    def forward(self, x, cond_embed, genre, times, interval, cond_drop_prob=0.2):

        # project to latent space
        x = self.input_projection(x)
        t = self.time_mlp(times * 1000)
        intv = self.intv_mlp(interval * 1000)
        t = t + intv
        cond = self.cond_projection(cond_embed)


        # Encode music conditioning.
        for i in range(self.music_num_layers):
            cond = self.musiclayerstack[i](cond, genre)

        # Decode motion.
        for i in range(self.dance_num_layers):
            x = self.dancelayerstack[i](x, cond, genre, t)

        # project to SMPL space
        output = self.final_layer(x)
        return output

    def infer_pred(self, x, cond_embed, genre, times, interval):
        return self.forward(x, cond_embed, genre, times, interval, cond_drop_prob=0.0)
