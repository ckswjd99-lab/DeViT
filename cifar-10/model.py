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
    def forward(self, x, tgt_mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=tgt_mask) + x
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

    def forward_masked(self, img):
        # custom mask: cls는 모든 토큰을 볼 수 있지만 나머지는 역삼각형인 특이한 mask
        # [cls token] 1 1 1 1 1 1 ...
        # [token # 0] 0 1 0 0 0 0 ...
        # [token # 1] 0 1 1 0 0 0 ...
        # [token # 2] 0 1 1 1 0 0 ...
        # [token # 3] 0 1 1 1 1 0 ...
        # [token # 4] 0 1 1 1 1 1 ...

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        custom_mask = torch.triu(torch.ones(n + 1, n + 1), diagonal=1).to(x.device)
        # fill the first column with 0
        custom_mask[:, 0] = 0
        # fill the first row with 1
        custom_mask[0, :] = 1
        

        x = self.transformer(x, custom_mask)

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
    def __init__(self, *, image_sizes=[32, 16, 8], patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
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
        self.order_shuffle_low = torch.Tensor([2, 3, 0, 1]).type(torch.int32) + 0
        self.order_shuffle_mid = torch.Tensor([14, 15,  6, 11,  5, 17, 10,  8, 18, 19, 12, 16,  7,  9,  4, 13]).type(torch.int32)
        self.order_shuffle_high = torch.Tensor(
            [65, 44, 67, 55, 80, 77, 58, 47, 61, 71, 34, 22, 63, 48, 28, 42, 75, 53,
            21, 64, 45, 29, 33, 43, 30, 46, 51, 73, 69, 56, 82, 50, 37, 25, 39, 62,
            68, 40, 59, 52, 32, 36, 20, 76, 79, 31, 66, 41, 23, 27, 60, 81, 38, 72,
            24, 49, 26, 83, 70, 54, 35, 57, 74, 78]
        ).type(torch.int32)

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
        num_patches_low = (self.image_sizes[2] // self.patch_size) ** 2
        num_patches_mid = (self.image_sizes[1] // self.patch_size) ** 2
        num_patches_high = (self.image_sizes[0] // self.patch_size) ** 2

        x[:, :num_patches_low] = x[:, self.order_shuffle_low]
        x[:, num_patches_low:num_patches_low + num_patches_mid] = x[:, self.order_shuffle_mid]
        x[:, num_patches_low + num_patches_mid:] = x[:, self.order_shuffle_high]

        # 마스킹: 하삼각형 마스킹
        tgt_mask = torch.tril(torch.ones(self.total_patches, self.total_patches)).to(x.device) != 0

        # Decoder 통과
        x = self.decoder(x, tgt_mask=tgt_mask)

        # Classifier 통과 (모든 feature 사용)
        output = self.mlp_head(x) # (batch_size, num_patches, num_classes)

        return output

    @torch.no_grad()
    def init_from_envit(self, envit_model):
        # EncoderViT의 Encoder의 가중치를 DecoderViT의 Decoder로 복사

        decoder = self.decoder.layers
        encoder = envit_model.transformer.layers

        self.mlp_head[1].weight.copy_(envit_model.mlp_head[1].weight)
        self.mlp_head[1].bias.copy_(envit_model.mlp_head[1].bias)

        for i, (decoder_block, encoder_block) in enumerate(zip(decoder, encoder)):
            decoder_block[0].norm.weight.copy_(encoder_block[0].norm.weight)
            decoder_block[0].norm.bias.copy_(encoder_block[0].norm.bias)
            decoder_block[0].fn.to_qkv.weight.copy_(encoder_block[0].fn.to_qkv.weight)
            decoder_block[0].fn.to_out[0].weight.copy_(encoder_block[0].fn.to_out[0].weight)
            decoder_block[0].fn.to_out[0].bias.copy_(encoder_block[0].fn.to_out[0].bias)

            decoder_block[1].norm.weight.copy_(encoder_block[1].norm.weight)
            decoder_block[1].norm.bias.copy_(encoder_block[1].norm.bias)
            decoder_block[1].fn.net[0].weight.copy_(encoder_block[1].fn.net[0].weight)
            decoder_block[1].fn.net[0].bias.copy_(encoder_block[1].fn.net[0].bias)
            decoder_block[1].fn.net[3].weight.copy_(encoder_block[1].fn.net[3].weight)
            decoder_block[1].fn.net[3].bias.copy_(encoder_block[1].fn.net[3].bias)

        # 위에서 복사한 가중치에 대해 requires_grad=False로 설정
        # for param in self.mlp_head[1].parameters():
        #     param.requires_grad = False

        # for block in decoder:
        #     for param in block[0].parameters():
        #         param.requires_grad = False
        #     for param in block[1].parameters():
        #         param.requires_grad = False


if __name__ == "__main__":
    PATCH_SIZE = 4
    IN_CHANNELS = 3
    EMBED_DIM = 512
    NUM_HEADS = 6
    NUM_LAYERS = 8
    FFN_DIM = 512
    DROPOUT = 0.1
    EMB_DROPOUT = 0.1
    NUM_CLASSES = 10

    DEVIT_PATH = "./saves/devit_best_85.70.pt"
    ENVIT_PATH = "./saves/evit_best_86.60.pt"

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    ckpt_devit = torch.load(DEVIT_PATH, weights_only=True)
    ckpt_envit = torch.load(ENVIT_PATH, weights_only=True)

    print(ckpt_devit.keys())

    model_devit = DecoderViT(
        image_sizes=[32, 16, 8], 
        patch_size=PATCH_SIZE, 
        num_classes=NUM_CLASSES, 
        dim=EMBED_DIM, 
        depth=NUM_LAYERS, 
        heads=NUM_HEADS, 
        mlp_dim=FFN_DIM, 
        channels=IN_CHANNELS, 
        dropout=DROPOUT, 
        emb_dropout=EMB_DROPOUT
    ).to(device=device)

    model_envit = EncoderViT(
        image_size=32, 
        patch_size=PATCH_SIZE, 
        num_classes=NUM_CLASSES, 
        dim=EMBED_DIM, 
        depth=NUM_LAYERS, 
        heads=NUM_HEADS, 
        mlp_dim=FFN_DIM, 
        channels=IN_CHANNELS, 
        dropout=DROPOUT, 
        emb_dropout=EMB_DROPOUT
    ).to(device=device)

    model_devit.load_state_dict(ckpt_devit["model_state_dict"])
    model_envit.load_state_dict(ckpt_envit["model_state_dict"])

    model_devit.eval()
    model_envit.eval()

    model_devit.init_from_envit(model_envit)
