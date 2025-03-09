from src.backbones import utae, unet3d, convlstm, convgru, fpn
from src.panoptic import paps
from src.backbones.swinutae import SwinUTAE  # Import the new SwinUTAE module


def get_model(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "swinutae":
            model = SwinUTAE(
                input_dim=10,
                encoder_dims=config.encoder_widths,
                decoder_dims=config.decoder_widths,
                out_conv=config.out_conv,
                window_size=config.window_size if (hasattr(config, "window_size") and isinstance(config.window_size, int)) else 7,
                input_resolution=(128, 128),
                num_heads=config.num_heads if hasattr(config, "num_heads") else [3, 6, 12, 24],
                mlp_ratio=config.mlp_ratio if hasattr(config, "mlp_ratio") else 4.0,
                qkv_bias=config.qkv_bias if hasattr(config, "qkv_bias") else True,
                qk_scale=config.qk_scale if hasattr(config, "qk_scale") else None,
                drop_rate=config.drop_rate if hasattr(config, "drop_rate") else 0.0,
                attn_drop_rate=config.attn_drop_rate if hasattr(config, "attn_drop_rate") else 0.0,
                drop_path_rate=config.drop_path_rate if hasattr(config, "drop_path_rate") else 0.1,
                agg_mode=config.agg_mode,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
            )
        elif config.model == "utae":
            model = utae.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.model == "unet3d":
            model = unet3d.UNet3D(
                in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=10,
                num_classes=config.num_classes,
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        elif config.model == "uconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        elif config.model == "buconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
    elif mode == "panoptic":
        if config.backbone == "swinutae":
            model = SwinUTAE(
                input_dim=10,
                encoder_dims=config.encoder_widths,
                decoder_dims=config.decoder_widths,
                out_conv=config.out_conv,
                window_size=config.window_size if hasattr(config, "window_size") else 7,
                input_resolution=(128, 128),
                num_heads=config.num_heads if hasattr(config, "num_heads") else [3, 6, 12, 24],
                mlp_ratio=config.mlp_ratio if hasattr(config, "mlp_ratio") else 4.0,
                qkv_bias=config.qkv_bias if hasattr(config, "qkv_bias") else True,
                qk_scale=config.qk_scale if hasattr(config, "qk_scale") else None,
                drop_rate=config.drop_rate if hasattr(config, "drop_rate") else 0.0,
                attn_drop_rate=config.attn_drop_rate if hasattr(config, "attn_drop_rate") else 0.0,
                drop_path_rate=config.drop_path_rate if hasattr(config, "drop_path_rate") else 0.1,
                agg_mode=config.agg_mode,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=True,
                return_maps=False,
                pad_value=config.pad_value,
            )
        elif config.backbone == "utae":
            model = utae.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=True,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.backbone == "uconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=True,
                padding_mode="zeros",
                pad_value=0,
            )
        else:
            raise NotImplementedError

        model = paps.PaPs(
            encoder=model,
            num_classes=config.num_classes,
            shape_size=config.shape_size,
            mask_conv=config.mask_conv,
            min_confidence=config.min_confidence,
            min_remain=config.min_remain,
            mask_threshold=config.mask_threshold,
        )
        return model
    else:
        raise NotImplementedError
