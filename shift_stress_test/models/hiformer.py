
import torch
import torch.nn as nn
import torchvision
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block

MODEL_NAME = "hiformer"
FAMILY = "hybrid_cnn_transformer"
SOURCE = {"type": "github", "repo": "https://github.com/amirhossein-kz/HiFormer", "commit": "6726d2845bed3c7f7e28156d90bed510b34a6f9b", "files": ["models/HiFormer.py", "models/Encoder.py", "models/Decoder.py", "utils.py", "configs/HiFormer_configs.py"], "variant": "HiFormer-B, no pretrained Swin/ResNet weights; MultiScaleBlock kept exactly as the official code"}


class ConfigDict(dict):

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        num_branches = len(dim)
        self.num_branches = num_branches
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))
        if len(self.blocks) == 0:
            self.blocks = None
        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d + 1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))
        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))
        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d + 1) % num_branches]), act_layer(), nn.Linear(dim[(d + 1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        inp = x
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(inp, self.projs)]
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], inp[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, inp[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        outs_b = [block(x_) for x_, block in zip(outs, self.blocks)]
        return outs

class Attention(nn.Module):

    def __init__(self, dim, factor, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim * factor), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SwinTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6], num_heads=[3, 6, 12], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):
        super().__init__()
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=None)
            self.layers.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

class PyramidFeatures(nn.Module):

    def __init__(self, config, img_size=224, in_channels=3):
        super().__init__()
        self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        resnet = getattr(torchvision.models, config.cnn_backbone)(weights=None)
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        self.p1_pm = PatchMerging((config.image_size // config.patch_size, config.image_size // config.patch_size), config.swin_pyramid_fm[0])
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p2_pm = PatchMerging((config.image_size // config.patch_size // 2, config.image_size // config.patch_size // 2), config.swin_pyramid_fm[1])
        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1)
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for i in range(5):
            x = self.resnet_layers[i](x)
        fm1 = x
        fm1_ch = self.p1_ch(x)
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped + sw1
        norm1 = self.norm_1(sw1_skipped)
        sw1_CLS = self.avgpool_1(norm1.transpose(1, 2))
        sw1_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw1_CLS)
        fm1_sw1 = self.p1_pm(sw1_skipped)
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        fm2 = self.p2(fm1)
        fm2_ch = self.p2_ch(fm2)
        fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        fm2_sw2_skipped = fm2_reshaped + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skipped)
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        fm3 = self.p3(fm2)
        fm3_ch = self.p3_ch(fm3)
        fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        fm3_sw3_skipped = fm3_reshaped + fm2_sw3
        norm2 = self.norm_2(fm3_sw3_skipped)
        sw3_CLS = self.avgpool_2(norm2.transpose(1, 2))
        sw3_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw3_CLS)
        return [torch.cat((sw1_CLS_reshaped, sw1_skipped), dim=1), torch.cat((sw3_CLS_reshaped, fm3_sw3_skipped), dim=1)]

class All2Cross(nn.Module):

    def __init__(self, config, img_size=224, in_chans=3, embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid = PyramidFeatures(config=config, img_size=img_size, in_channels=in_chans)
        n_p1 = (config.image_size // config.patch_size) ** 2
        n_p2 = (config.image_size // config.patch_size // 4) ** 2
        num_patches = (n_p1, n_p2)
        self.num_branches = 2
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale, drop=config.drop_rate, attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)
        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        xs = self.pyramid(x)
        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] += self.pos_embed[i]
        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        return xs

class ConvUpsample(nn.Module):

    def __init__(self, in_chans=384, out_chans=[128], upsample=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_tower = nn.ModuleList()
        for i, out_ch in enumerate(self.out_chans):
            if i > 0:
                self.in_chans = out_ch
            self.conv_tower.append(nn.Conv2d(self.in_chans, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv_tower.append(nn.GroupNorm(32, out_ch))
            self.conv_tower.append(nn.ReLU(inplace=False))
            if upsample:
                self.conv_tower.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.convs_level = nn.Sequential(*self.conv_tower)

    def forward(self, x):
        return self.convs_level(x)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)

class HiFormer(nn.Module):

    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)
        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)
        self.segmentation_head = SegmentationHead(in_channels=16, out_channels=n_classes, kernel_size=3)
        self.conv_pred = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange('b (h w) d -> b d h w', h=self.img_size // self.patch_size[i], w=self.img_size // self.patch_size[i])(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            reshaped_embed.append(embed)
        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)
        out = self.segmentation_head(C)
        return out

def hiformer_b_config(n_classes, img_size):
    cfg = ConfigDict()
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = img_size
    cfg.patch_size = 4
    cfg.num_classes = n_classes
    cfg.swin_pretrained_path = None
    cfg.cnn_backbone = 'resnet50'
    cfg.cnn_pyramid_fm = [256, 512, 1024]
    cfg.resnet_pretrained = False
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio = (2.0, 2.0, 1.0)
    cfg.drop_rate = 0.0
    cfg.attn_drop_rate = 0.0
    cfg.drop_path_rate = 0.0
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True
    return cfg


def build_model(num_classes=2, img_size=224):
    return HiFormer(hiformer_b_config(num_classes, img_size), img_size=img_size, in_chans=3, n_classes=num_classes)
