

import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinTAE(nn.Module):
    def __init__(self, in_channels=128, embed_dim=128, window_size=8, return_att=True, depth=2, num_heads=8):
        super().__init__()
        self.return_att = return_att
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.depth = depth
        self.num_heads = num_heads

        self.swin_cache = {}  # to store Swin models for different T
        self.out_proj = nn.Linear(embed_dim, in_channels)

    def get_swin_for_T(self, T):
        if T not in self.swin_cache:
            swin = SwinTransformer(
                img_size=(T, 1),
                patch_size=1,
                in_chans=self.in_channels,
                num_classes=0,
                embed_dim=self.embed_dim,
                depths=(self.depth,),
                num_heads=(self.num_heads,),
                window_size=self.window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True
            )
            self.swin_cache[T] = swin.to(next(self.parameters()).device)
        return self.swin_cache[T]

    def forward(self, x, batch_positions=None, pad_mask=None):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, T, C]
        x = x.view(B * H * W, T, C)  # [B*H*W, T, C]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B*H*W, C, T, 1]

        swin = self.get_swin_for_T(T)
        x = swin(x)  # [B*H*W, embed_dim]

        x = self.out_proj(x)  # [B*H*W, C]
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        dummy_attn = torch.ones((self.num_heads, B, T, H, W), device=x.device)
        return x, dummy_attn


# #code 2
# import torch
# import torch.nn as nn
# from timm.models.swin_transformer import SwinTransformer

# class SwinTAE(nn.Module):
#     def __init__(self, in_channels=128, embed_dim=128, window_size=8, return_att=True, depth=2, num_heads=8):
#         super().__init__()
#         self.return_att = return_att
#         self.in_channels = in_channels

#         # Treat temporal dimension as spatial height (T x 1 image)
#         self.swin = SwinTransformer(
#             img_size=(64, 1),     # T x 1 patch (you may adjust 64 to your max temporal length)
#             patch_size=1,         # process every timestep individually
#             in_chans=in_channels, # number of input features per timestep
#             num_classes=0,
#             embed_dim=embed_dim,
#             depths=(depth,),      # only one stage
#             num_heads=(num_heads,),
#             window_size=window_size,
#             mlp_ratio=4.,
#             qkv_bias=True,
#             drop_rate=0.0,
#             attn_drop_rate=0.0,
#             drop_path_rate=0.1,
#             ape=False,
#             patch_norm=True
#         )

#         # Project output back to original channel space if needed
#         self.out_proj = nn.Linear(embed_dim, in_channels)

#     def forward(self, x, batch_positions=None, pad_mask=None):
#         # x: [B, T, C, H, W]
#         B, T, C, H, W = x.shape
#         x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, T, C]
#         x = x.view(B * H * W, T, C)  # Treat each spatial pixel as a sequence: [B*H*W, T, C]

#         # Reshape to Swin expected input: [B', C, T, 1]
#         x = x.permute(0, 2, 1).unsqueeze(-1)  # [B*H*W, C, T, 1]

#         # Run Swin Transformer
#         x = self.swin(x)  # output shape: [B*H*W, embed_dim]

#         # Project and reshape back
#         x = self.out_proj(x)  # [B*H*W, C]
#         x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

#         # Return fake attention map (if needed for compatibility)
#         dummy_attn = torch.ones((num_heads, B, T, H, W), device=x.device)
#         return x, dummy_attn



# code 1
# import torch
# import torch.nn as nn
# from timm.models.swin_transformer import SwinTransformerBlock

# class SwinTAE(nn.Module):
#     def __init__(self, in_channels=128, embed_dim=128, depth=2, num_heads=8, window_size=8, return_att=True):
#         super().__init__()
#         self.return_att = return_att
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.window_size = window_size

#         self.input_proj = nn.Linear(in_channels, embed_dim)
#         self.swin_blocks = nn.Sequential(
#             *[SwinTransformerBlock(
#                 dim=embed_dim,
#                 input_resolution=(64, 1),  # <--- required by older timm
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else window_size // 2,
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 norm_layer=nn.LayerNorm,
#             ) for i in range(depth)]
#         )
#         self.output_proj = nn.Linear(embed_dim, in_channels)

#     def forward(self, x, batch_positions=None, pad_mask=None):
#         # x: [B, T, C, H, W]
#         B, T, C, H, W = x.shape
#         x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, T, C]
#         x = x.view(B * H * W, T, C)  # [B*H*W, T, C]
#         x = self.input_proj(x)

#         for blk in self.swin_blocks:
#             x = blk(x)

#         x = self.output_proj(x)  # [B*H*W, T, C]
#         x = x.mean(dim=1)  # temporal average pooling: [B*H*W, C]
#         x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

#         dummy_attn = torch.ones((8, B, T, H, W), device=x.device)  # fake attention
#         return x, dummy_attn
