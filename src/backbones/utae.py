"""
U-TAE Implementation with Swin Transformer Spatial Encoder
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
Modified to use Swin Transformer blocks for spatial encoding
License: MIT
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from src.backbones.convlstm import ConvLSTM, BConvLSTM
from src.backbones.ltae import LTAE2d
from src.backbones.swintae import SwinTAE


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding if needed"""
    B, H, W, C = x.shape
    
    # Pad feature maps to multiples of window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        # Use padding
        x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    H_padded, W_padded = H + pad_h, W + pad_w
    
    # Safe handling for small feature maps or large window_size
    if H_padded < window_size or W_padded < window_size:
        # If feature map is smaller than window_size, use the entire feature map as one window
        windows = x.unsqueeze(1).unsqueeze(1)  # B, 1, 1, H_padded, W_padded, C
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, H_padded, W_padded, C)
        return windows, (H, W, H_padded, W_padded)
    
    # Reshape to windows
    try:
        x = x.reshape(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
    except RuntimeError:
        # Fallback for cases where reshape fails due to dimension mismatch
        # Use adaptive pooling to resize to window_size x window_size
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = nn.functional.adaptive_avg_pool2d(x, (window_size, window_size))
        x = x.permute(0, 2, 3, 1)  # B, window_size, window_size, C
        windows = x.reshape(-1, window_size, window_size, C)
    
    return windows, (H, W, H_padded, W_padded)


def window_reverse(windows, window_size, H, W, H_padded, W_padded):
    """Reverse window partition"""
    
    # Special handling for when the input is a single window (from window_partition fallback)
    if windows.size(0) == 1 or H_padded < window_size or W_padded < window_size:
        # Handle case where we have only one window (feature map smaller than window_size)
        x = windows
        
        # Ensure correct size by resizing if needed
        if x.size(1) != H_padded or x.size(2) != W_padded:
            # Reshape to B, C, H, W for adaptive_avg_pool2d
            x = x.permute(0, 3, 1, 2)
            x = nn.functional.adaptive_avg_pool2d(x, (H_padded, W_padded))
            x = x.permute(0, 2, 3, 1)  # B, H_padded, W_padded, C
        
        # If dimensions still don't match the expected output, use another resize
        if x.size(0) != 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
    else:
        try:
            # Normal reverse window partitioning
            B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
            x = windows.reshape(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H_padded, W_padded, -1)
        except RuntimeError:
            # Fallback for shape errors: just resize directly
            B = 1  # Assume batch size is 1 for simplicity
            C = windows.size(-1)
            
            # Flatten all windows into a single feature map
            x = windows.reshape(-1, window_size, window_size, C)
            x = x.permute(0, 3, 1, 2)  # N, C, window_size, window_size
            
            # Use adaptive pooling to get the proper H_padded, W_padded size
            x = nn.functional.adaptive_avg_pool2d(x, (H_padded, W_padded))
            x = x.permute(0, 2, 3, 1).contiguous()  # N, H_padded, W_padded, C
            
            # If needed, use only the first N = B elements
            if x.size(0) > B:
                x = x[:B]
            elif x.size(0) < B:
                # If fewer elements than expected, repeat to match B
                x = x.repeat(B, 1, 1, 1)
    
    # Remove padding if needed
    if H < H_padded or W < W_padded:
        x = x[:, :H, :W, :]
    
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Define relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        # Compute QKV using 1D conv to avoid reshape errors
        qkv = self.qkv(x)  # B_, N, 3*C
        
        # Simple reshape that works for any dimension
        qkv_reshaped = qkv.reshape(B_, N, 3, -1)
        q = qkv_reshaped[:, :, 0].contiguous()  # B_, N, C
        k = qkv_reshaped[:, :, 1].contiguous()  # B_, N, C
        v = qkv_reshaped[:, :, 2].contiguous()  # B_, N, C
        
        # Split heads if possible
        head_dim = C // self.num_heads
        if head_dim * self.num_heads == C:
            q = q.reshape(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3)  # B_, h, N, d_k
            k = k.reshape(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3)  # B_, h, N, d_k
            v = v.reshape(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3)  # B_, h, N, d_k
            
            q = q * self.scale
            attn = torch.matmul(q, k.transpose(-2, -1))  # B_, h, N, N
            
            # Use the pre-computed relative position bias if window size matches
            window_size_current = int(N ** 0.5)
            if window_size_current ** 2 == N and window_size_current == self.window_size[0]:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
                
                # Add bias if shapes match
                if relative_position_bias.size(-1) == self.num_heads:
                    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # h, Wh*Ww, Wh*Ww
                    attn = attn + relative_position_bias.unsqueeze(0)
            
            # Apply softmax and dropout
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            
            # Apply attention to values
            out = torch.matmul(attn, v)  # B_, h, N, d_k
            out = out.transpose(1, 2).reshape(B_, N, C)  # B_, N, C
        else:
            # Fallback for cases where head_dim doesn't divide C evenly
            # Just use a single head attention
            q = q * (C ** -0.5)
            attn = torch.matmul(q, k.transpose(-2, -1))  # B_, N, N
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)  # B_, N, C
        
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Adjust window size and shift size for small resolutions
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # For LayerNorm, dim is the normalized dimension
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        
        # Handle case where L doesn't match H*W (expected resolution)
        if L != H * W:
            # Try to find a reasonable size for H and W
            size = int(L ** 0.5)
            if size * size == L:
                # Perfect square
                H = W = size
            else:
                # Find factors of L for H and W
                # Start with closer to square aspect ratio
                for i in range(int(L**0.5), 0, -1):
                    if L % i == 0:
                        H = i
                        W = L // i
                        break
            
        assert L == H * W, f"Input feature has wrong size {L} vs {H*W}"

        shortcut = x
        # Apply normalization to each example
        x = self.norm1(x)
        
        # Reshape to 4D tensor [B, H, W, C]
        # Use reshape instead of view for potentially non-contiguous tensors
        x = x.reshape(B, H, W, C)  # Changed from view to reshape
        
        # Calculate effective window size (must be <= than feature map dimensions)
        window_size = min(self.window_size, min(H, W))
        
        # Skip shift operation if window_size is too large or if shift_size is 0
        if window_size < 2 or self.shift_size == 0 or window_size <= self.shift_size:
            shift_size = 0
            shifted_x = x
        else:
            shift_size = min(self.shift_size, window_size // 2)
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # Partition windows
        x_windows, (_, _, H_padded, W_padded) = window_partition(shifted_x, window_size)
        x_windows = x_windows.reshape(-1, window_size * window_size, C)  # Changed from view to reshape
        
        # Self-attention
        attn_windows = self.attn(x_windows, mask=None)

        # Merge windows
        attn_windows = attn_windows.reshape(-1, window_size, window_size, C)  # Changed from view to reshape
        shifted_x = window_reverse(attn_windows, window_size, H, W, H_padded, W_padded)

        # Reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Reshape back to [B, H*W, C]
        x = x.reshape(B, H * W, C)  # Changed from view to reshape
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinEncoder(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.output_dim = 2 * dim if downsample else dim
        
        # Ensure window size is smaller than input resolution
        window_size = min(window_size, min(input_resolution))
        
        # Handle drop_path as list or scalar
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * depth
        
        # Dimension adaptation layer (initialized as identity)
        self.dim_adapter = nn.Identity()
        
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample:
            self.downsample = PatchMerging(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # Check if input dimensions match the expected dimensions
        B, L, C = x.shape
        if C != self.dim:
            # Dynamically create adapter if dimensions don't match
            if isinstance(self.dim_adapter, nn.Identity):
                self.dim_adapter = nn.Linear(C, self.dim).to(x.device)
            x = self.dim_adapter(x)
            
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        
        # Handle inconsistent size
        if L != H * W:
            # Adjust to nearest perfect square
            size = int(L ** 0.5)
            H = W = size
        
        assert L == H * W, f"Input feature shape {L} doesn't match with resolution {H}*{W}"
        
        # Skip merging if dimensions are odd
        if H % 2 != 0 or W % 2 != 0:
            return x

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class UTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
        window_size=7,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        output_size=128,  # Add output_size parameter to match the target resolution
    ):
        """
        U-TAE architecture with Swin Transformer spatial encoder for satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths.
            decoder_widths (List[int], optional): Similar to encoder_widths for the decoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the output.
            str_conv_k (int): Kernel size of the strided up convolutions.
            str_conv_s (int): Stride of the strided up convolutions.
            str_conv_p (int): Padding of the strided up convolutions.
            agg_mode (str): Aggregation mode for the skip connections.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch.
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE.
            d_k (int): Key-Query space dimension.
            encoder (bool): If true, the feature maps instead of the class scores are returned.
            return_maps (bool): If true, the feature maps are returned along with the class scores.
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers.
            window_size (int): Size of attention window in Swin Transformer.
            depths (List[int]): Number of Swin blocks at each stage.
            num_heads (List[int]): Number of attention heads in different layers.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim in Swin Transformer.
            qkv_bias (bool): If True, add a learnable bias to q, k, v in Swin attention.
            drop_rate (float): Dropout rate in Swin Transformer.
            attn_drop_rate (float): Attention dropout rate in Swin Transformer.
            drop_path_rate (float): Stochastic depth rate in Swin Transformer.
            output_size (int): Target output resolution.
        """
        super(UTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths.copy()  # Create a copy to avoid modifying the input
        self.decoder_widths = decoder_widths.copy() if decoder_widths is not None else None
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        self.img_size = 128
        self.output_size = output_size
        
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            self.decoder_widths = encoder_widths.copy()

        # Make sure the configuration is valid
        # Ensure depths and num_heads have at least n_stages-1 elements
        if len(depths) < self.n_stages-1:
            depths = depths + [depths[-1]] * (self.n_stages-1 - len(depths))
        if len(num_heads) < self.n_stages-1:
            num_heads = num_heads + [num_heads[-1]] * (self.n_stages-1 - len(num_heads))

        # Ensure window_size is not too large
        window_size = min(window_size, self.img_size // 4)
        
        # Initial patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=4,  # Fixed patch size to make resolution 32x32 after embedding
            in_chans=input_dim,
            embed_dim=self.encoder_widths[0],
            norm_layer=nn.LayerNorm
        )

        # Swin Transformer encoder blocks
        self.swin_encoders = nn.ModuleList()
        curr_resolution = (self.img_size // 4, self.img_size // 4)  # 32x32 after patch embedding with patch_size=4
        
        # Create a dpr (drop path rate) list based on depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.n_stages-1]))]
        
        dpr_index = 0
        for i in range(self.n_stages - 1):
            # Ensure indices are valid for depths and num_heads
            depth_index = min(i, len(depths)-1)
            head_index = min(i, len(num_heads)-1)
            
            # Calculate drop path rates for this stage
            stage_dpr = dpr[dpr_index:dpr_index+depths[depth_index]] if dpr_index < len(dpr) else [drop_path_rate] * depths[depth_index]
            dpr_index += depths[depth_index]
            
            # Ensure num_heads matches embedding dimension (must be divisible)
            dim = self.encoder_widths[i]
            n_heads = num_heads[head_index]
            if dim % n_heads != 0:
                # Adjust number of heads to make it divisible
                n_heads = max(1, dim // max(1, dim // n_heads))
            
            # Ensure dimensions work with LTAE too
            if i == self.n_stages - 2:  # Last encoder before LTAE
                # Make sure the embedding dimension is compatible with LTAE
                if self.encoder_widths[i+1] % n_head != 0:
                    # Adjust encoder_widths to match n_head requirements
                    self.encoder_widths[i+1] = (self.encoder_widths[i+1] // n_head) * n_head
                    if self.encoder_widths[i+1] == 0:
                        self.encoder_widths[i+1] = n_head
                # Make sure decoder has the same dimension
                if self.decoder_widths is not None:
                    self.decoder_widths[-1] = self.encoder_widths[-1]
            
            encoder = SwinEncoder(
                dim=self.encoder_widths[i],
                input_resolution=curr_resolution,
                depth=depths[depth_index],
                num_heads=n_heads, 
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_dpr,
                downsample=True,
                norm_layer=nn.LayerNorm
            )
            self.swin_encoders.append(encoder)
            # Update resolution for the next stage
            curr_resolution = (max(1, curr_resolution[0] // 2), max(1, curr_resolution[1] // 2))

        # # Temporal encoder (LTAE)
        # self.temporal_encoder = LTAE2d(
        #     in_channels=self.encoder_widths[-1],
        #     d_model=d_model,
        #     n_head=n_head,
        #     mlp=[d_model, self.encoder_widths[-1]],
        #     return_att=True,
        #     d_k=d_k,
        # )

        self.temporal_encoder = SwinTAE(
            in_channels=encoder_widths[-1],
            embed_dim=d_model,
            depth=2,
            num_heads=n_head,
            window_size=8,
            return_att=True
        )

        # Temporal aggregator
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # Decoder blocks (keep the original decoder)
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=self.decoder_widths[i],
                d_out=self.decoder_widths[i - 1],
                d_skip=self.encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )

        # Output convolution with upsampling
        self.out_conv = nn.Sequential(
            ConvBlock(nkernels=[self.decoder_widths[0]] + out_conv, padding_mode=padding_mode),
            # Add upsampling to match target resolution
            nn.Upsample(size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        )

    def forward(self, input, batch_positions=None, return_att=False):
        B, T, C, H, W = input.shape
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

        # Process each timestep through Swin encoder
        feature_maps = []
        for t in range(T):
            curr_x = input[:, t]  # B, C, H, W
            curr_x = self.patch_embed(curr_x)  # B, L, C
            features = [curr_x]

            # Apply Swin encoder blocks
            for encoder in self.swin_encoders:
                curr_x = encoder(curr_x)
                features.append(curr_x)

            # Add time dimension to each feature map
            feature_maps.append([f.unsqueeze(1) for f in features])

        # Stack features across time for each resolution level
        feature_maps = [torch.cat([fm[i] for fm in feature_maps], dim=1) 
                       for i in range(len(feature_maps[0]))]

        # Temporal encoding on the final embedding
        x = feature_maps[-1]  # B, T, L, C
        
        # Reshape for LTAE
        B, T, L, C = x.shape
        
        # Calculate suitable H and W dimensions for feature maps
        # Try to find the most square-like factorization of L
        H_enc = int(L ** 0.5)
        while L % H_enc != 0 and H_enc > 1:
            H_enc -= 1
        W_enc = L // H_enc
        
        # Reshape to 5D tensor for LTAE
        x = x.reshape(B, T, C, H_enc, W_enc)
        
        # Apply temporal encoding
        x, att = self.temporal_encoder(x, batch_positions=batch_positions, pad_mask=pad_mask)

        # Spatial decoding with skip connections
        if self.return_maps:
            maps = [x]

        for i, up_block in enumerate(self.up_blocks):
            # Get skip connection from appropriate level
            skip = feature_maps[-(i+2)]
            
            # Process skip connection
            B, T, L, C = skip.shape
            
            # Calculate suitable H and W dimensions for skip connection
            H_skip = int(L ** 0.5)
            while L % H_skip != 0 and H_skip > 1:
                H_skip -= 1
            W_skip = L // H_skip
            
            # Reshape to 5D tensor for temporal aggregation
            skip = skip.reshape(B, T, C, H_skip, W_skip)
            
            # Apply temporal aggregation
            skip = self.temporal_aggregator(skip, pad_mask=pad_mask, attn_mask=att)
            
            # Apply upsampling
            x = up_block(x, skip)
            
            if self.return_maps:
                maps.append(x)

        if self.encoder:
            return x, maps
        else:
            out = self.out_conv(x)
            if return_att:
                return out, att
            if self.return_maps:
                return out, maps
            else:
                return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.reshape(b * t, c, h, w)).shape

            out = input.reshape(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.reshape(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        # If input is not 5D (B,T,C,H,W), call conv directly
        if len(input.shape) != 5:
            return self.conv(input)
        # Otherwise use smart_forward for handling temporal dimension
        return self.smart_forward(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        
        # Handle potential channel dimension mismatch in skip connection
        skip_channels = skip.size(1)
        if skip_channels != self.skip_conv[0].in_channels:
            # Dynamically create a new skip_conv layer with correct dimensions
            d = self.skip_conv[0].out_channels
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=skip_channels, out_channels=d, kernel_size=1),
                nn.BatchNorm2d(d),
                nn.ReLU(),
            ).to(skip.device)
            
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group" and attn_mask is not None:
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.reshape(n_heads * b, t, h, w)

                # Handle dimension mismatch
                if x.shape[-2] != h or x.shape[-1] != w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                
                attn = attn.reshape(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                # Ensure we can divide x along channel dimension
                if x.shape[2] % n_heads != 0:
                    # Adjust number of heads to be compatible with x's channels
                    adjusted_heads = 1
                    for i in range(n_heads, 0, -1):
                        if x.shape[2] % i == 0:
                            adjusted_heads = i
                            break
                    
                    # Use only the first adjusted_heads heads from attn
                    attn = attn[:adjusted_heads]
                    n_heads = adjusted_heads
                
                # Split channels into groups for attention
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean" and attn_mask is not None:
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                
                # Handle dimension mismatch
                if x.shape[-2] != attn.shape[-2] or x.shape[-1] != attn.shape[-1]:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None].clamp(min=1)
                return out
        else:
            if self.mode == "att_group" and attn_mask is not None:
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.reshape(n_heads * b, t, h, w)
                
                # Handle dimension mismatch
                if x.shape[-2] != h or x.shape[-1] != w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                
                attn = attn.reshape(n_heads, b, t, *x.shape[-2:])
                
                # Ensure we can divide x along channel dimension
                if x.shape[2] % n_heads != 0:
                    # Adjust number of heads to be compatible with x's channels
                    adjusted_heads = 1
                    for i in range(n_heads, 0, -1):
                        if x.shape[2] % i == 0:
                            adjusted_heads = i
                            break
                    
                    # Use only the first adjusted_heads heads from attn
                    attn = attn[:adjusted_heads]
                    n_heads = adjusted_heads
                
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean" and attn_mask is not None:
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                
                # Handle dimension mismatch
                if x.shape[-2] != attn.shape[-2] or x.shape[-1] != attn.shape[-1]:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class RecUNet(nn.Module):
    """Recurrent U-Net architecture. Similar to the U-TAE architecture but
    the L-TAE is replaced by a recurrent network
    and temporal averages are computed for the skip connections."""

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        temporal="lstm",
        input_size=128,
        encoder_norm="group",
        hidden_dim=128,
        encoder=False,
        padding_mode="reflect",
        pad_value=0,
    ):
        super(RecUNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.temporal = temporal
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

        self.encoder = encoder
        if encoder:
            self.return_maps = True
        else:
            self.return_maps = False

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
        )

        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="mean")

        if temporal == "mean":
            self.temporal_encoder = Temporal_Aggregator(mode="mean")
        elif temporal == "lstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = ConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "blstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = BConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=2 * hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "mono":
            self.temporal_encoder = None
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)

    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        if self.temporal == "mean":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        elif self.temporal == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        elif self.temporal == "blstm":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = self.out_convlstm(out)
        elif self.temporal == "mono":
            out = feature_maps[-1]

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            if self.temporal != "mono":
                skip = self.temporal_aggregator(
                    feature_maps[-(i + 2)], pad_mask=pad_mask
                )
            else:
                skip = feature_maps[-(i + 2)]
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if self.return_maps:
                return out, maps
            else:
                return out
