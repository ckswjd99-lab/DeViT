import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Attention 모듈에서 마스킹 적용
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EncoderViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, tgt_mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=tgt_mask) + x
            x = ff(x) + x
        return x

class DecoderViT(nn.Module):
    def __init__(self, *, image_sizes, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.image_sizes = image_sizes
        self.patch_size = patch_size
        self.num_classes = num_classes

        # 패치 임베딩 레이어
        self.patch_embeds = nn.ModuleList([nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size * patch_size, dim),
        ) for _ in image_sizes])

        # Positional Encoding (학습 가능)
        self.pos_embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, dim)) for img_size in image_sizes])

        self.dropout = nn.Dropout(emb_dropout)

        # Decoder
        self.decoder = TransformerDecoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Classifier (단순화를 위해 Linear 레이어 사용)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # 전체 패치 개수 계산
        self.total_patches = sum([(img_size // patch_size) ** 2 for img_size in self.image_sizes])

        # 각 해상도 안에서 섞을 순서 결정
        self.order_shuffle_low = torch.randperm(4)
        self.order_shuffle_mid = torch.randperm(16) + 4
        self.order_shuffle_high = torch.randperm(64) + 20

    def forward(self, pyramid):
        # pyramid는 Image Pyramid (3단계: 8x8x3, 16x16x3, 32x32x3)를 담은 리스트
        embeddings = []

        # 각 해상도 별 임베딩 계산 및 positional encoding 추가
        for i, image in enumerate(pyramid):
            embed = self.patch_embeds[i](image)
            pos_embed = self.pos_embeddings[i]

            # interpolate positional embeddings if needed
            if embed.shape[1] != pos_embed.shape[1]:
                pos_embed = F.interpolate(pos_embed.permute(0, 2, 1), size=embed.shape[1], mode='linear', align_corners=False).permute(0, 2, 1)

            embed += pos_embed
            embeddings.append(embed)

        # 패치 임베딩 결합
        x = torch.cat(embeddings, dim=1)
        x = self.dropout(x)

        # 각 해상도 안에서 중앙이 더 먼저 오도록
        x[:, :4, :] = x[:, self.order_shuffle_low, :]
        x[:, 4:20, :] = x[:, self.order_shuffle_mid, :]
        x[:, 20:, :] = x[:, self.order_shuffle_high, :]

        # 마스킹: 하삼각형 마스킹
        tgt_mask = torch.tril(torch.ones(self.total_patches, self.total_patches)).to(x.device) != 0

        # Decoder 통과
        x = self.decoder(x, tgt_mask=tgt_mask)

        # Classifier 통과 (모든 feature 사용)
        output = self.mlp_head(x) # (batch_size, num_patches, num_classes)

        return output

