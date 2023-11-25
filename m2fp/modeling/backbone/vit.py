from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from fairscale.nn.checkpoint import checkpoint_wrapper

from .utils import *

try:
    import xformers.ops as xops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            use_rel_pos=False,
            input_size=None,
            xformers=True,
            use_lora=False,
            lora_info=dict,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            xformers (bool): If True, use xops.memory_efficient_attention to replace naive attention.
            use_lora (bool): If True, use LoRALinear replacing nn.Linear when generating qkv.
            lora_info (dict): Dictionary for storing LoRA information.
        """
        super().__init__()
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        self.xformers = xformers if XFORMERS_IS_AVAILBLE else False

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if not use_lora:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = LoRALinear(dim, dim * 3, bias=qkv_bias,
                                  lora_alpha=lora_info["lora_alpha"], lora_dim=lora_info["lora_dim"], qkv=True)
        self.proj = nn.Linear(dim, dim)

        if self.use_rel_pos:
            assert (
                    (input_size is not None) and (not xformers)
            ), "Input size must be provided and xformers must be turn off if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x, H, W):
        B, N, C = x.shape

        if self.xformers:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
            q, k, v = qkv.unbind(0)  # [B, N, nHead, C // nHead]
            x = xops.memory_efficient_attention(q, k, v, attn_bias=None).reshape(B, N, -1)
        else:  # vitdet impl.
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)  # [B * nHead, N, C // nHead]
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            attn = attn.softmax(dim=-1)
            x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.0,
            init_values=None,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            swiglu=False,
            use_rel_pos=False,
            input_size=None,
            xformers=True,
            use_lora=False,
            lora_info=dict,
            use_tome=False,
            tome_info=dict,
            use_repadapter=False,
            repadapter_info=dict,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            init_values (float or None): Ratio of layer scale.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            swiglu (bool): If True, use SwiGLUFFNFused replacing MLP.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            xformers (bool): If True, use xops.memory_efficient_attention to replace naive attention.
            use_lora (bool): If True, use LoRALinear replacing nn.Linear when generating qkv.
            lora_info (dict): Dictionary for storing LoRA information.
            use_tome (bool): If True, use token merging before self-attention.
            tome_info (dict): Dictionary for storing ToMe information.
            use_repadapter (bool): If True, use RepAdapter before attention and MLP linear layer.
            repadapter_info (dict): Dictionary for storing RepAdapter information.
        """
        super().__init__()
        self.xformers = xformers if XFORMERS_IS_AVAILBLE else False
        self.use_tome = use_tome
        self.tome_info = tome_info

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size,
            xformers=xformers,
            use_lora=use_lora,
            lora_info=lora_info,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if swiglu:
            self.mlp = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.use_repadapter = use_repadapter
        if use_repadapter:
            adapter_kwarts = {
                'hidden_dim': repadapter_info['adapter_hidden_dim'],
                'scale': repadapter_info['adapter_scale'],
                'groups': repadapter_info['adapter_groups'],
                'dropout': repadapter_info['adapter_dropout'],
            }
            self.adapter_mlp = RepAdapter(dim, **adapter_kwarts)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W):
        B, _, C = x.shape

        shortcut = x
        x = self.norm1(x)  # [B, 1124, 768]

        # task & image tokens self-attention
        if not self.use_tome:
            x = self.attn(x, H, W)  # [B, 1124, 768]
        else:
            m_a, u_a = compute_merge(x, H, W, self.tome_info)
            x = m_a(x)
            x = self.attn(x, H, W)  # [B, 612, 768]
            x = u_a(x)

        if self.gamma_1 is None:
            x = shortcut + self.drop_path(x)
            if self.use_repadapter:
                x = x + self.drop_path(self.mlp(self.adapter_mlp(self.norm2(x))))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = shortcut + self.drop_path(self.gamma_1 * x)
            if self.use_repadapter:
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.adapter_mlp(self.norm2(x))))
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
            self,
            # backbone
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            init_values=None,
            norm_pre_=False,
            norm_post=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            swiglu=False,
            use_abs_pos=True,
            use_rel_pos=False,
            use_act_checkpoint=False,
            pretrain_img_size=224,
            pretrain_use_cls_token=True,
            xformers=True,
            freeze=False,
            # lora
            lora_info=dict,
            # tome
            tome_info=dict,
            # repadapter
            repadapter_info=dict,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            init_values (float or None): Ratio of layer scale.
            norm_pre_ (bool): If True, add a norm layer before transformer blocks.
            norm_post (bool): If True, add a norm layer after transformer blocks.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            swiglu (bool): If True, use SwiGLUFFNFused replacing MLP.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretraining models use class token.
            xformers (bool): If True, use xops.memory_efficient_attention to replace naive attention.
            freeze (bool): If True, freezing the backbone parameters, exclude query_feat.
            common_stride (int): Stride of the transformer features.
            lora_info (dict): Dictionary for storing LoRA information.
            tome_info (dict): Dictionary for storing ToMe information.
            repadapter_info (dict): Dictionary for storing RepAdapter information.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.norm_pre_ = norm_pre_
        self.norm_post = norm_post

        # patch embedding (projection)
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # absolute position embedding
        if use_abs_pos:
            # initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # normalization before transformer encoder
        if self.norm_pre_:
            self.norm_pre = norm_layer(embed_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # transformer encoder
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                init_values=init_values,
                norm_layer=norm_layer,
                act_layer=act_layer,
                swiglu=swiglu,
                use_rel_pos=use_rel_pos,
                input_size=(img_size // patch_size, img_size // patch_size),
                xformers=xformers,
                use_lora=i in lora_info["lora_block_indexes"],
                lora_info=lora_info,
                use_tome=i in tome_info["merge_attn_indexes"],
                tome_info=tome_info,
                use_repadapter=i in repadapter_info['adapter_block_indexes'],
                repadapter_info=repadapter_info,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        # normalization after transformer encoder
        if self.norm_post:
            self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

        if freeze:
            self._freeze_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_backbone(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        if self.norm_pre_:
            for param in self.norm_pre.parameters():
                param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        if self.norm_post:
            for param in self.norm.parameters():
                param.requires_grad = False

        for module_name, module in self.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if isinstance(module, LoRALinear) and "lora" in module_param_name:
                    value.requires_grad = True
                if isinstance(module, RepAdapter) or 'adapter_mlp' in module_name:
                    value.requires_grad = True

    def forward(self, x):
        # feature shape example for vit-base with 512x512 input size
        B, _, _, _ = x.size()  # [B, 3, 512, 512]

        x, (Hp, Wp) = self.patch_embed(x)  # BCHW -> BNC [B, 1024, 768]; (Hp, Wp) = (32, 32)

        if self.pos_embed is not None:
            x = x + resize_pos_embed(self.pos_embed, 1 if self.pretrain_use_cls_token else 0, (Hp, Wp))

        if self.norm_pre_:
            x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            x = blk(x, Hp, Wp)

        if self.norm_post:
            x = self.norm(x)

        x = x.contiguous().view(B, Hp, Wp, -1).permute(0, 3, 1, 2)  # [B, 32, 32, 768] --> [B, 768, 32, 32]

        return x


@BACKBONE_REGISTRY.register()
class ViTSFP(ViT, Backbone):
    def __init__(self, cfg, input_shape):
        img_size = cfg.MODEL.VIT.IMG_SIZE
        patch_size = cfg.MODEL.VIT.PATCH_SIZE
        in_chans = 3
        embed_dim = cfg.MODEL.VIT.EMBED_DIM
        depth = cfg.MODEL.VIT.DEPTH
        num_heads = cfg.MODEL.VIT.NUM_HEADS
        mlp_ratio = cfg.MODEL.VIT.MLP_RATIO
        qkv_bias = cfg.MODEL.VIT.QKV_BIAS
        drop_path_rate = cfg.MODEL.VIT.DROP_PATH_RATE
        init_values = cfg.MODEL.VIT.INIT_VALUES
        norm_pre_ = cfg.MODEL.VIT.NORM_PRE
        norm_post = cfg.MODEL.VIT.NORM_POST
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        swiglu = cfg.MODEL.VIT.SWIGLU
        use_abs_pos = cfg.MODEL.VIT.USE_ABS_POS
        use_rel_pos = cfg.MODEL.VIT.USE_REL_POS
        use_act_checkpoint = cfg.MODEL.VIT.USE_ACT_CHECKPOINT
        pretrain_img_size = cfg.MODEL.VIT.PRETRAIN_IMG_SIZE
        pretrain_use_cls_token = cfg.MODEL.VIT.PRETRAIN_USE_CLS_TOKEN
        xformers = cfg.MODEL.VIT.XFORMERS
        freeze = cfg.MODEL.VIT.FREEZE

        lora_alpha = cfg.MODEL.LORA.LORA_ALPHA
        lora_dim = cfg.MODEL.LORA.LORA_DIM
        lora_block_indexes = cfg.MODEL.LORA.LORA_BLOCK_INDEXES

        ratio = cfg.MODEL.TOME.RATIO
        sx = cfg.MODEL.TOME.SX
        sy = cfg.MODEL.TOME.SY
        use_rand = cfg.MODEL.TOME.USE_RAND
        merge_attn_indexes = cfg.MODEL.TOME.MERGE_ATTN_INDEXES

        adapter_hidden_dim = cfg.MODEL.REP_ADAPTER.ADAPTER_HIDDEN_DIM
        adapter_scale = cfg.MODEL.REP_ADAPTER.ADAPTER_SCALE
        adapter_groups = cfg.MODEL.REP_ADAPTER.ADAPTER_GROUPS
        adapter_dropout = cfg.MODEL.REP_ADAPTER.ADAPTER_DROPOUT
        adapter_block_indexes = cfg.MODEL.REP_ADAPTER.ADAPTER_BLOCK_INDEXES

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            norm_pre_=norm_pre_,
            norm_post=norm_post,
            norm_layer=norm_layer,
            act_layer=act_layer,
            swiglu=swiglu,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            use_act_checkpoint=use_act_checkpoint,
            pretrain_img_size=pretrain_img_size,
            pretrain_use_cls_token=pretrain_use_cls_token,
            xformers=xformers,
            freeze=freeze,
            # lora
            lora_info={
                "lora_alpha": lora_alpha,
                "lora_dim": lora_dim,
                "lora_block_indexes": lora_block_indexes,
            },
            # tome
            tome_info={
                "ratio": ratio,
                "sx": sx,
                "sy": sy,
                "use_rand": use_rand,
                "merge_attn_indexes": merge_attn_indexes,
            },
            # RepAdapter
            repadapter_info={
                "adapter_hidden_dim": adapter_hidden_dim,
                "adapter_scale": adapter_scale,
                "adapter_groups": adapter_groups,
                "adapter_dropout": adapter_dropout,
                "adapter_block_indexes": adapter_block_indexes,
            }
        )

        sfp_dim = cfg.MODEL.SFP.SFP_DIM
        scale_factors = cfg.MODEL.SFP.SCALE_FACTORS
        self.sfp = SimpleFeaturePyramid(
            in_channels=embed_dim,
            out_channels=sfp_dim,  # 256
            scale_factors=scale_factors,  # [4.0, 2.0, 1.0, 0.5]
            norm="LN",  # "LN"
        )

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": sfp_dim,
            "res3": sfp_dim,
            "res4": sfp_dim,
            "res5": sfp_dim,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """

        assert (
                x.dim() == 4
        ), f"VisionTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        x = super().forward(x)
        x = self.sfp(x)

        return x

