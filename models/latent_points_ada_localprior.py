# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch 
from loguru import logger 
import torch.nn as nn 
import torch.nn.functional as F
from .latent_points_ada import PVCNN2Unet 
from .utils import mask_inactive_variables 

# diffusion model for latent points 
class PVCNN2Prior(PVCNN2Unet): 
    sa_blocks = [ # conv_configs, sa_configs # conv_configs (out_channels, num_blocks, voxel_resolution), sa_configs (num_centers, radius, num_neighbors, out_channels)
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 128))),
        (None, (16, 0.8, 32, (128, 128, 128))), 
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)), # fp_configs, conv_configs
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, args, num_input_channels, cfg):

        # only cfg is used 
        self.clip_forge_enable = cfg.clipforge.enable
        clip_forge_dim = cfg.clipforge.feat_dim
        num_input_channels = num_classes = cfg.shapelatent.latent_dim + cfg.ddpm.input_dim 
        self.num_classes = num_classes 
        embed_dim = cfg.ddpm.time_dim 
        use_att = True 
        extra_feature_channels = cfg.shapelatent.latent_dim 
        self.num_points = cfg.data.tr_max_sample_points 
        dropout = cfg.ddpm.dropout 
        time_emb_scales = cfg.sde.embedding_scale  # 1k default 
        part_type_channels = cfg.data.num_parts    # 4
        self.part_type_channels = part_type_channels
        concat_part_type = cfg.latent_pts.concat_part_type_ddpm
        self.concat_part_type = concat_part_type
        self.pred_part_type = cfg.latent_pts.pred_part_type
        if self.concat_part_type:
            extra_feature_channels += part_type_channels    # change the input dim.
        logger.info('[Build Prior Model] nclass={}, embed_dim={}, use_att={},'
                'extra_feature_channels={}, dropout={}, time_emb_scales={} num_point={}'
                'concat_part_type={}, part_type_dim={}',
                num_classes, embed_dim, use_att, extra_feature_channels, dropout, time_emb_scales,
                self.num_points, concat_part_type, part_type_channels)
        use_att_interleaving = True
        # Attention: we are not using time_emb_scales here, but the embedding_scale
        super().__init__(
                num_classes, embed_dim, use_att, dropout=dropout,
                input_dim=cfg.ddpm.input_dim,
                extra_feature_channels=extra_feature_channels, 
                time_emb_scales=time_emb_scales,
                verbose=True,
                condition_input=False, 
                cfg=cfg,
                sa_blocks=self.sa_blocks,
                fp_blocks=self.fp_blocks, 
                clip_forge_enable=self.clip_forge_enable, clip_forge_dim=clip_forge_dim,
                use_att_interleaving=use_att_interleaving,
                concat_part_type=concat_part_type,
                part_type_channels=part_type_channels,
                pred_part_type=self.pred_part_type) 
        # init mixing logit 
        self.mixed_prediction = cfg.sde.mixed_prediction  # This enables mixed prediction
        if self.mixed_prediction:
            logger.info('init-mixing_logit = {}, after sigmoid = {}', 
                    cfg.sde.mixing_logit_init, torch.sigmoid(torch.tensor(cfg.sde.mixing_logit_init))
                    )
            init = cfg.sde.mixing_logit_init * torch.ones(size=[1, num_input_channels*self.num_points, 1, 1])
            self.mixing_logit = torch.nn.Parameter(init, requires_grad=True)
            self.is_active = None
        else: # no mixing_logit
          self.mixing_logit = None
          self.is_active = None

    def forward(self, x, t, *args, **kwargs):
        # x: [B, N*4(8192), 1, 1], t: [B], condition_input: [B, z(128), 1, 1]
        # require shape for x: B,C,N
        ## CHECKEQ(x.shape[-1], self.num_classes)
        assert('condition_input' in kwargs), 'require condition_input'
        if self.mixed_prediction and self.is_active is not None:
            x = mask_inactive_variables(x, self.is_active)
        input_shape = x.shape 
        if self.pred_part_type:
            input_shape = list(input_shape)
            input_shape[1] += self.part_type_channels * self.num_points
        x = x.view(-1,self.num_points,self.num_classes).permute(0,2,1).contiguous() # [B, 4, N]
        B = x.shape[0] 
        if self.concat_part_type:
            out = super().forward(x, t=t, style=kwargs['condition_input'].squeeze(-1).squeeze(-1), \
                                clip_feat=kwargs.get('clip_feat', None), part_types=kwargs.get('part_types', None)) # [B, 4, N(2048)]
        else:
            out = super().forward(x, t=t, style=kwargs['condition_input'].squeeze(-1).squeeze(-1), \
                                clip_feat=kwargs.get('clip_feat', None)) # [B, 4, N(2048)]
        return out.permute(0,2,1).contiguous().view(input_shape)  # [B, N*4, 1, 1]
        # -1,self.num_classes) # BDN -> BND -> BN,D
