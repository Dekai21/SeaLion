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
import numpy as np
from .pvcnn2_ada import \
        create_pointnet2_sa_components, create_pointnet2_fp_modules, LinearAttention, create_mlp_components, SharedMLP 
import copy

# the building block of encode and decoder for VAE 

class PVCNN2Unet(nn.Module):
    """
        copied and modified from https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/model/pvcnn_generation.py#L172 
    """
    def __init__(self, 
                 num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, 
                 input_dim=3,
                 width_multiplier=1, 
                 voxel_resolution_multiplier=1,
                 time_emb_scales=1.0,
                 verbose=True, 
                 condition_input=False, 
                 point_as_feat=1, cfg={}, 
                 sa_blocks={}, fp_blocks={}, 
                 clip_forge_enable=0,
                 clip_forge_dim=512,
                 use_att_interleaving=False,
                 concat_part_type=False,
                 part_type_channels=4,
                 pred_part_type=False,
                 ):
        super().__init__()
        logger.info('[Build Unet] extra_feature_channels={}, input_dim={}',
                extra_feature_channels, input_dim)
        self.input_dim = input_dim 

        self.clip_forge_enable = clip_forge_enable 
        self.sa_blocks = sa_blocks 
        self.fp_blocks = fp_blocks
        self.point_as_feat = point_as_feat
        self.condition_input = condition_input
        assert extra_feature_channels >= 0
        self.time_emb_scales = time_emb_scales
        self.embed_dim = embed_dim
        self.pred_part_type = pred_part_type
        ## assert(self.embed_dim == 0)
        if self.embed_dim > 0: # has time embedding 
            # for prior model, we have time embedding, for VAE model, no time embedding 
            self.embedf = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        if self.clip_forge_enable:
            self.clip_forge_mapping = nn.Linear(clip_forge_dim, embed_dim)  # clip_forge_dim, embed_dim: 512, 0
            style_dim = cfg.latent_pts.style_dim    # 128
            self.style_clip = nn.Linear(style_dim + embed_dim, style_dim) 

        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = \
            create_pointnet2_sa_components(
            input_dim=input_dim,
            sa_blocks=self.sa_blocks, 
            extra_feature_channels=extra_feature_channels, 
            with_se=True, 
            embed_dim=embed_dim, # time embedding dim 
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, 
            voxel_resolution_multiplier=voxel_resolution_multiplier, 
            verbose=verbose, cfg=cfg, num_classes=num_classes,
            concat_part_type=concat_part_type,
            part_type_channels=part_type_channels
        )   # [[PVConv, PVConv, PointNetSAModule], [PVConv, PointNetSAModule], [PVConv, PointNetSAModule], [PointNetSAModule]], [3, 64, 128, 256], 128
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else LinearAttention(channels_sa_features, 8, verbose=verbose)
        self.global_att_interleaving = None if not use_att_interleaving else nn.ModuleList([LinearAttention(sa_in_channels[1], 8, verbose=verbose), LinearAttention(sa_in_channels[2], 8, verbose=verbose), LinearAttention(sa_in_channels[3], 8, verbose=verbose)])

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels + input_dim - 3
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, 
            sa_in_channels=sa_in_channels, 
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            verbose=verbose, cfg=cfg 
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        if pred_part_type:
            fp_layers_seg, channels_fp_features = create_pointnet2_fp_modules(
                fp_blocks=self.fp_blocks, in_channels=channels_sa_features, 
                sa_in_channels=sa_in_channels, 
                with_se=True, embed_dim=embed_dim,
                use_att=use_att, dropout=dropout,
                width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
                verbose=verbose, cfg=cfg 
            )
            self.fp_layers_seg = nn.ModuleList(fp_layers_seg)

        if not pred_part_type:
            layers, _ = create_mlp_components(
                    in_channels=channels_fp_features, 
                    out_channels=[128, dropout, num_classes], # was 0.5
                    # out_channels=[128, dropout, num_classes + part_type_channels], # was 0.5 # dk: TODO: num_classes + num_part_types
                    classifier=True, dim=2, width_multiplier=width_multiplier,
                    cfg=cfg)
            self.classifier = nn.ModuleList(layers)
        else:
            layers, _ = create_mlp_components(
                    in_channels=channels_fp_features, 
                    out_channels=[128, dropout, num_classes],
                    classifier=True, dim=2, width_multiplier=width_multiplier,
                    cfg=cfg)
            self.classifier = nn.ModuleList(layers)
            layers_seg, _ = create_mlp_components(
                    in_channels=channels_fp_features, 
                    out_channels=[128, dropout, part_type_channels],
                    classifier=True, dim=2, width_multiplier=width_multiplier,
                    cfg=cfg)
            if pred_part_type:
                layers_seg.append(nn.Softmax(dim=1))
            self.classifier_seg = nn.ModuleList(layers_seg)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:,0]
        assert(len(timesteps.shape) == 1), f'get shape: {timesteps.shape}'  
        timesteps = timesteps * self.time_emb_scales 

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, **kwargs):
        # when used in encoding (inputs are points & z0), inputs.shape = [B, 3, Npoints], style.shape = [B, z(128)]
        # when used in diffusion, input: [B, 4, N], 4 := (input_dim + latent_dim), kwargs['style']: [B, z(128)], t: [B]
        B = inputs.shape[0]
        coords = inputs[:, :self.input_dim, :].contiguous() # [B, 3, N]
        features = inputs 
        temb = kwargs.get('t', None)    # [B]
        if temb is not None:
            t = temb 
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            temb =  self.embedf(self.get_timestep_embedding(t, inputs.device 
                ))[:,:,None].expand(-1,-1,inputs.shape[-1])
            temb_ori = temb  # [B, emb(64), N]
        
        style = kwargs['style'] # [B, z(128)]
        if self.clip_forge_enable:
            clip_feat = kwargs['clip_feat'] 
            assert(clip_feat is not None), f'require clip_feat as input'
            clip_feat = self.clip_forge_mapping(clip_feat)  # [B, N] ->    x [B, embed_dim]
            style = torch.cat([style, clip_feat], dim=1).contiguous()
            style = self.style_clip(style)  # [B, z+embed_dim] -> [B, z]

        if 'part_types' in kwargs:
            # part_seg = copy.deepcopy(kwargs['part_types'])
            # part_seg -= 8   # [B, N]
            # # part_seg = torch.eye(4)[part_seg.cpu().data.numpy(),]   # [B, N, 4]
            # part_seg = torch.cat([torch.eye(4), torch.zeros(1, 4)], dim=0)[part_seg.cpu().data.numpy(),]   # [B, N, 4]
            part_seg = kwargs['part_types']
            part_seg = part_seg.transpose(1, 2).to(inputs.device)
            features = torch.concat([features, part_seg], dim=1).to(features.device)    # part_seg: [B, part(4), N]

        if 'normals' in kwargs:
            normals = kwargs['normals'] # [B, N, 3]
            normals = normals.transpose(1, 2).to(inputs.device)

        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers): # 4 layers
            in_features_list.append(features)
            coords_list.append(coords)
            if i > 0 and temb is not None:
                #TODO: implement a sa_blocks forward function; check if is PVConv layer and kwargs get grid_emb, take as additional input 
                features = torch.cat([features, temb], dim=1)
                if 'part_types' in kwargs:
                    if 'normals' in kwargs:
                        features, coords, temb, _, part_seg, normals = sa_blocks((features, coords, temb, style, part_seg, normals))    
                    else:
                        features, coords, temb, _, part_seg = sa_blocks((features, coords, temb, style, part_seg))
                    if i == 1 or i == 2:
                        # TODO: dk: implement a self.global_att here.
                        if self.global_att_interleaving is not None:
                            features = self.global_att_interleaving[i](features)
                else:
                    features, coords, temb, _ = sa_blocks((features, coords, temb, style)) 
            else: # i == 0 or temb is None 
                if 'part_types' in kwargs:
                    if 'normals' in kwargs:
                        features, coords, temb, _, part_seg, normals = sa_blocks((features, coords, temb, style, part_seg, normals))    
                    else:
                        features, coords, temb, _, part_seg = sa_blocks((features, coords, temb, style, part_seg))
                    if temb is not None and self.global_att_interleaving is not None:
                        features = self.global_att_interleaving[i](features)
                else:
                    features, coords, temb, _ = sa_blocks((features, coords, temb, style))

        # in_features_list[0] = inputs[:, 3:, :].contiguous() # TODO: dk: if the input dim = 3, if this step will be skipped? why the 3d coordinate is neglected here?
        in_features_list[0] = in_features_list[0][:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)    # [B, D(128), N'(16)] -> [B, D(128), N'(16)]

        if self.pred_part_type:
            # features_seg = features.clone()
            # temb_seg = temb.clone()
            # coords_seg = coords.clone()
            # features_seg = features.detach().clone()
            # temb_seg = temb.detach().clone()
            # coords_seg = coords.detach().clone()
            # style_seg = style.detach().clone()
            features_seg = features.clone()
            temb_seg = temb.clone()
            coords_seg = coords.clone()
            style_seg = style.clone()
            features_seg.requires_grad_(True)
            temb_seg.requires_grad_(True)
            coords_seg.requires_grad_(True)
            style_seg.requires_grad_(True)
            in_features_list_seg = [i.clone() for i in in_features_list]
            coords_list_seg = [i.clone() for i in coords_list]
            for i in in_features_list_seg:
                i.requires_grad_(True)
            for i in coords_list_seg:
                i.requires_grad_(True)
            for fp_idx, fp_blocks  in enumerate(self.fp_layers_seg):    # feature propagation? feature: [B, 128, 16] -> [B, 64, 2048]
                if temb_seg is not None:
                    features_seg, coords_seg, temb_seg, _ = fp_blocks((
                        coords_list_seg[-1-fp_idx], coords_seg, 
                        torch.cat([features_seg,temb_seg],dim=1), 
                        in_features_list_seg[-1-fp_idx], temb_seg, style_seg))
                else:
                    features_seg, coords_seg, temb_seg, _ = fp_blocks((
                        coords_list_seg[-1-fp_idx], coords_seg, 
                        features_seg, 
                        in_features_list_seg[-1-fp_idx], temb_seg, style_seg))

        for fp_idx, fp_blocks  in enumerate(self.fp_layers):    # feature propagation? feature: [B, 128, 16] -> [B, 64, 2048]
            if temb is not None:
                features, coords, temb, _ = fp_blocks((
                    coords_list[-1-fp_idx], coords, 
                    torch.cat([features,temb],dim=1), 
                    in_features_list[-1-fp_idx], temb, style))
            else:
                features, coords, temb, _ = fp_blocks((
                    coords_list[-1-fp_idx], coords, 
                    features, 
                    in_features_list[-1-fp_idx], temb, style))

        if self.pred_part_type:
            # features_seg = features.clone()
            for l in self.classifier_seg:
                if isinstance(l, SharedMLP):
                    features_seg = l(features_seg, style_seg)
                else:
                    features_seg = l(features_seg)
        for l in self.classifier:
            if isinstance(l, SharedMLP):
                features = l(features, style)
            else:
                features = l(features)
        if self.pred_part_type:
            features = torch.cat([features, features_seg], dim=1)
        return features # [B, 4, 2048]

class PointTransPVC(nn.Module):
    # encoder : B,N,3 -> B,N,2*D 
    sa_blocks = [ # conv_configs (out_channels, num_blocks, voxel_resolution), sa_configs (num_centers, radius, num_neighbors, out_channels)
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (128, 128, 128))), 
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)), # fp_configs, conv_configs
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, zdim, input_dim, args={}):
        super().__init__()
        self.zdim = zdim 
        part_type_channels = args.data.num_parts    # 4
        concat_part_type = args.latent_pts.concat_part_type
        logger.info('[Build Enc] point_dim={}, context_dim={}, concat_part_type={}, part_type_dim={}', input_dim, zdim, concat_part_type, part_type_channels)
        self.layers = PVCNN2Unet(2*zdim+input_dim*2, 
                embed_dim=0, use_att=1, extra_feature_channels=0 + part_type_channels,
                input_dim=args.ddpm.input_dim, cfg=args,
                sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks,
                dropout=args.ddpm.dropout,
                use_att_interleaving=True,
                concat_part_type=concat_part_type,
                part_type_channels=part_type_channels)
        self.skip_weight = args.latent_pts.skip_weight
        self.pts_sigma_offset = args.latent_pts.pts_sigma_offset 
        self.input_dim = input_dim  # 3

    def forward(self, inputs, **kwargs):
        x, style = inputs # [B, Npoints, 3], [B, D1(128)]
        B,N,D = x.shape 
        output = self.layers(x.permute(0,2,1).contiguous(), style=style, **kwargs).permute(0,2,1).contiguous() # [B,N,D(8)]  [mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, mu_z, sigma_z]

        pt_mu_1d = output[:,:,:self.input_dim].contiguous() # [B, N, 3]
        pt_sigma_1d = output[:,:,self.input_dim:2*self.input_dim].contiguous() - self.pts_sigma_offset  # [B, N, 3]
        
        pt_mu_1d = self.skip_weight * pt_mu_1d + x 
        if self.zdim > 0:
            ft_mu_1d = output[:,:,2*self.input_dim:-self.zdim].contiguous() # [B, N, 1]
            ft_sigma_1d = output[:,:,-self.zdim:].contiguous()

            mu_1d = torch.cat([pt_mu_1d, ft_mu_1d], dim=2).view(B,-1).contiguous()  # [B, N4]?
            sigma_1d = torch.cat([pt_sigma_1d, ft_sigma_1d], dim=2).view(B,-1).contiguous() 
        else:
            mu_1d = pt_mu_1d.view(B,-1).contiguous()
            sigma_1d = pt_sigma_1d.view(B,-1).contiguous() 
        return {'mu_1d': mu_1d, 'sigma_1d': sigma_1d}   # [B, N4], [B, N4]

class LatentPointDecPVC(nn.Module):
    """ input x: [B,Npoint,D] with [B,Npoint,3] 
    """
    sa_blocks = [ # conv_configs, sa_configs
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (128, 128, 128))), 
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)), # fp_configs, conv_configs
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, point_dim, context_dim, num_points=None, args={}, **kwargs):
        super().__init__()
        self.point_dim = point_dim  
        part_type_channels = args.data.num_parts    # 4
        concat_part_type = args.latent_pts.concat_part_type
        logger.info('[Build Dec] point_dim={}, context_dim={}, concat_part_type={}, part_type_dim={}', point_dim, context_dim, concat_part_type, part_type_channels)
        self.context_dim  = context_dim + self.point_dim 
        # self.num_points = num_points
        if num_points is None:
            self.num_points = args.data.tr_max_sample_points
        else:
            self.num_points = num_points
        self.layers = PVCNN2Unet(point_dim, embed_dim=0, use_att=1, 
                extra_feature_channels=context_dim + part_type_channels,
                input_dim=args.ddpm.input_dim, cfg=args, 
                sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks, 
                dropout=args.ddpm.dropout,
                use_att_interleaving=True,
                concat_part_type=concat_part_type,
                part_type_channels=part_type_channels)
        self.skip_weight = args.latent_pts.skip_weight

    def forward(self, x, beta, context, style, **kwargs):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d). [not used] 
            beta:     Time. (B, ). [not used] 
            context:  Latent points, (B,N_pts*D_latent_pts), D_latent_pts = D_input + D_extra, [B, N*4]
            style: Shape latents. [B, z]
        Returns: 
            points: (B,N,3)
        """ 

        # CHECKDIM(context, 1, self.num_points*self.context_dim)
        assert(context.shape[1] == self.num_points*self.context_dim)
        context = context.view(-1,self.num_points,self.context_dim) # BND 
        x = context[:,:,:self.point_dim]    # [B, N, 3]
        output = self.layers(context.permute(0,2,1).contiguous(), style=style, **kwargs).permute(0,2,1).contiguous() # [B, N, 3]
        output = output * self.skip_weight + x 
        return output  

