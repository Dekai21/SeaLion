# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" to train hierarchical VAE model with 2 prior 
one for style latent, one for latent pts, 
based on trainers/train_prior.py 
"""
import os
import time
from PIL import Image
import gc
import functools
import psutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from loguru import logger
import torch.distributed as dist
from torch import optim
from utils.ema import EMA
from utils.model_helper import import_model, loss_fn
from utils.vis_helper import visualize_point_clouds_3d
from utils.eval_helper import compute_NLL_metric 
from utils import model_helper, exp_helper, data_helper
from utils.data_helper import normalize_point_clouds
## from utils.diffusion_discretized import DiffusionDiscretized
from utils.diffusion_pvd import DiffusionDiscretized
from utils.diffusion_continuous import make_diffusion, DiffusionBase
from utils.checker import *
from utils import utils
from matplotlib import pyplot as plt
# import third_party.pvcnn.functional as pvcnn_fn
from timeit import default_timer as timer
from torch.optim import Adam as FusedAdam
from torch.cuda.amp import autocast, GradScaler
from trainers.train_prior import Trainer as PriorTrainer
from trainers.train_prior import validate_inspect  # import Trainer as PriorTrainer

quiet = int(os.environ.get('quiet', 0))
VIS_LATENT_PTS = 0


@torch.no_grad()
def generate_samples_vada_2prior(shape, dae, diffusion, vae, num_samples, enable_autocast,
                                 ode_eps=0.00001, ode_solver_tol=1e-5,  # None,
                                 ode_sample=False, prior_var=1.0, temp=1.0, vae_temp=1.0, noise=None, need_denoise=False,
                                 ddim_step=0, clip_feat=None, cls_emb=None, ddim_skip_type='uniform', ddim_kappa=1.0,
                                 part_types=None):
    output = {}
    #kwargs = {}
    # if cls_emb is not None:
    #    kwargs['cls_emb'] = cls_emb
    if ode_sample == 1:
        assert isinstance(
            diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
        assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
        assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
        start = timer()
        condition_input = None
        eps_list = []
        for i in range(len(dae)):
            assert(cls_emb is None), f' not support yet'
            eps, nfe, time_ode_solve = diffusion.sample_model_ode(
                dae[i], num_samples, shape[i], ode_eps, ode_solver_tol, enable_autocast, temp, noise,
                condition_input=condition_input, clip_feat=clip_feat,
            )
            condition_input = eps
            eps_list.append(eps)
            output['sampled_eps'] = eps
        eps = vae.compose_eps(eps_list)  # torch.cat(eps, dim=1)

    elif ode_sample == 0:
        assert isinstance(
            diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
        assert noise is None, 'Noise is not used in ancestral sampling.'
        nfe = diffusion._diffusion_steps
        time_ode_solve = 999.999  # Yeah I know...
        start = timer()
        condition_input = None if cls_emb is None else cls_emb
        all_eps = []
        for i in range(len(dae)):
            if ddim_step > 0:
                assert(cls_emb is None), f'not support yet'
                eps, eps_list = diffusion.run_ddim(dae[i],
                                                   num_samples, shape[i], temp, enable_autocast,
                                                   is_image=False, prior_var=prior_var, ddim_step=ddim_step,
                                                   condition_input=condition_input, clip_feat=clip_feat,
                                                   skip_type=ddim_skip_type, kappa=ddim_kappa)
            else:
                eps, eps_list = diffusion.run_denoising_diffusion(dae[i],
                                                                  num_samples, shape[i], temp, enable_autocast,
                                                                  is_image=False, prior_var=prior_var,
                                                                  condition_input=condition_input, clip_feat=clip_feat,
                                                                  part_types=part_types)
            condition_input = eps

            if cls_emb is not None:
                condition_input = torch.cat([condition_input,
                                             cls_emb.unsqueeze(-1).unsqueeze(-1)], dim=1)
            if i == 0:
                condition_input = vae.global2style(condition_input)
            # exit()
            all_eps.append(eps) # [[B, z, 1, 1], [B, N*4, 1, 1]]

            output['sampled_eps'] = eps
        eps = vae.compose_eps(all_eps)  # [B, z+N*4, 1, 1]
        output['eps_list'] = eps_list
    output['print/sample_mean_global'] = eps.view(
        num_samples, -1).mean(-1).mean()
    output['print/sample_var_global'] = eps.view(
        num_samples, -1).var(-1).mean()
    decomposed_eps = vae.decompose_eps(eps)
    # part_types = F.one_hot(decomposed_eps[1].reshape(-1, 2048, 8)[:, :, 4:].argmax(-1)).float() # FIXME
    if 'pred_types' in eps_list:
        pred_types = eps_list['pred_types']
    else:
        assert (part_types == 0).all(), 'part_types should be 0 for unconditional sampling'
        pred_types = part_types[:num_samples]   # proper shape of zero padding for unconditonal sampling
    kwargs = {'part_types': pred_types}
    image = vae.sample(num_samples=num_samples,
                       decomposed_eps=decomposed_eps, cls_emb=cls_emb, **kwargs)  # [B, N, 3]

    end = timer()
    sampling_time = end - start
    # average over GPUs
    nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
    sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
    time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
    return image, nfe_torch, time_ode_solve_torch, sampling_time_torch, output


class Trainer(PriorTrainer):
    is_diffusion = 0

    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        super().__init__(cfg, args)
        self.fun_generate_samples_vada = functools.partial(
            generate_samples_vada_2prior, ode_eps=cfg.sde.ode_eps,
            ddim_skip_type=cfg.sde.ddim_skip_type,
            ddim_kappa=cfg.sde.ddim_kappa)
        self.mIOU_list = {}

    def compute_loss_vae(self, tr_pts, global_step, **kwargs):
        """ compute forward for VAE model, used in global-only prior training 
        Input: 
            tr_pts: points, [B, N, 3]
            global_step: int 
        Returns: 
            output dict including entry: 
            'eps': z ~ posterior, [B, D1+N*4, 1, 1]
            'q_loss': 0 if not train vae else the KL+rec 
            'x_0_pred': global points if not train vae, [B, N, 3]
            'x_0_target': target points, [B, N, 3]

        """
        vae = self.model
        dae = self.dae
        args = self.cfg.sde
        distributed = args.distributed
        vae_sn_calculator = self.vae_sn_calculator
        num_total_iter = self.num_total_iter
        ## diffusion = self.diffusion_cont if self.cfg.sde.ode_sample else self.diffusion_disc
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        elif self.cfg.sde.ode_sample == 2:
            raise NotImplementedError
            # diffusion = [self.diffusion_cont, self.diffusion_disc]

        B = tr_pts.size(0)
        with torch.set_grad_enabled(args.train_vae):
            with autocast(enabled=args.autocast_train):
                # posterior and likelihood
                if not args.train_vae:
                    output = {}
                    all_eps, all_log_q, latent_list = vae.encode(tr_pts, **kwargs)    # [B, D1+N*4]
                    x_0_pred = x_0_target = tr_pts
                    vae_recon_loss = 0
                    def make_4d(x): return x.unsqueeze(-1).unsqueeze(-1) if \
                        len(x.shape) == 2 else x.unsqueeze(-1)
                    eps = make_4d(all_eps)  # [B, D1+N*4, 1, 1]
                    output.update({'eps': eps, 'q_loss': torch.zeros(1),
                                   'x_0_pred': tr_pts, 'x_0_target': tr_pts,
                                   'x_0': tr_pts, 'final_pred': tr_pts})
                else:
                    raise NotImplementedError
        return output
    
    @staticmethod
    def save_latent_pts(pcd, latent_pts, part_types_gt, part_types_pred, save_path):
        """"
        Args:
            pcd: [B, N, 3]
            latent_pts: [B, N, 4]
            part_types_gt: [B, N, part]
            part_types_pred: [B, N, part]
            save_path: str
        """
        save_dict = {'pcd': pcd, 'latent_pts': latent_pts, 'part_types_gt': part_types_gt, 'part_types_pred': part_types_pred}
        import pickle
        with open(f'{save_path}/latent_pts_airplane_042901.pt', 'wb') as fp:  # e.g. dd_122501_car_593_010_300_122701_rot_pseudo.pt
            pickle.dump(save_dict, fp)
        exit()

    # ------------------------------------------- #
    #   training fun                              #
    # ------------------------------------------- #

    def train_iter(self, data, *args, **kwargs):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        see get_loss in models/shapelatent_diffusion.py 
        """
        # some variables

        input_dim = self.cfg.ddpm.input_dim
        loss_type = self.cfg.ddpm.loss_type
        vae = self.model
        dae = self.dae
        dae.train()
        diffusion = self.diffusion_cont if self.cfg.sde.ode_sample else self.diffusion_disc
        if self.cfg.sde.ode_sample == 1:
            diffusion = self.diffusion_cont
        elif self.cfg.sde.ode_sample == 0:
            diffusion = self.diffusion_disc
        elif self.cfg.sde.ode_sample == 2:
            raise NotImplementedError  # not support training with different solver
            ## diffusion = [self.diffusion_cont, self.diffusion_disc]

        dae_optimizer = self.dae_optimizer
        vae_optimizer = self.vae_optimizer
        args = self.cfg.sde
        device = torch.device(self.device_str)
        num_total_iter = self.num_total_iter
        distributed = self.args.distributed
        dae_sn_calculator = self.dae_sn_calculator  # signal-to-noise ratio?
        vae_sn_calculator = self.vae_sn_calculator
        grad_scalar = self.grad_scalar

        global_step = step = kwargs.get('step', None)
        no_update = kwargs.get('no_update', False)

        # update_lr
        warmup_iters = len(self.train_loader) * args.warmup_epochs
        utils.update_lr(args, global_step, warmup_iters,
                        dae_optimizer, vae_optimizer)

        # input
        tr_pts = data['tr_points'].to(device)  # (B, Npoints, 3)
        inputs = data['input_pts'].to(  # [B, Npoints, 3]
            device) if 'input_pts' in data else None  # the noisy points
        tr_img = data['tr_img'].to(device) if 'tr_img' in data else None    # None
        model_kwargs = {}
        if self.cfg.data.cond_on_cat:
            class_label_int = data['cate_idx'].view(-1)  # .to(device)
            nclass = self.cfg.data.nclass
            class_label = torch.nn.functional.one_hot(class_label_int, nclass)
            model_kwargs['class_label'] = class_label.float().to(device)
        if 'part_types' in data:
            model_kwargs['part_types'] = data['part_types'].to(device)
            data['part_types'] = data['part_types'].to(device)
        if 'seg_mask' in data:
            data['seg_mask'] = data['seg_mask'].view(-1).bool().to(device)
        if 'pred_seg_weights' in data:
            data['pred_seg_weights'] = data['pred_seg_weights'].to(device)

        B = batch_size = tr_pts.size(0)
        if tr_img is not None:
            # tr_img: B,nimg,3,H,W
            # logger.info('image: {}', tr_img.shape)
            nimg = tr_img.shape[1]
            tr_img = tr_img.view(B*nimg, *tr_img.shape[2:])
            clip_feat = self.clip_model.encode_image(
                tr_img).view(B, nimg, -1).mean(1).float()
        else:
            clip_feat = None    # dk: clip_feat = data['part_types']
        if self.cfg.clipforge.enable:
            assert(clip_feat is not None)

        # optimize vae params
        vae_optimizer.zero_grad()
        output = self.compute_loss_vae(
            # tr_pts, global_step, inputs=inputs, **model_kwargs)
            tr_pts, global_step, **model_kwargs)

        # the interface between VAE and DAE is eps.
        eps = output['eps'].detach()  # [B, D1+N*4, 1, 1]
        CHECK4D(eps)
        dae_kwarg = {}
        if self.cfg.data.cond_on_cat:
            dae_kwarg['condition_input'] = output['cls_emb']
        dae_kwarg['part_types'] = data['part_types']

        # train prior
        if args.train_dae:
            dae_optimizer.zero_grad()
            with autocast(enabled=args.autocast_train):
                # get diffusion quantities for p sampling scheme and reweighting for q
                t_p, var_t_p, m_t_p, obj_weight_t_p, _, g2_t_p = \
                    diffusion.iw_quantities(B, args.time_eps,   # [B]
                                            args.iw_sample_p, args.iw_subvp_like_vp_sde)
                # logger.info('t_p: {}, var: {}, m_t: {}', t_p[0], var_t_p[0], m_t_p[0])
                # t_p(timestep): [B], var_t_p: [B, 1, 1, 1], m_t_p: [B, 1, 1, 1]; x_t = m_t_p * x_0 + var_t_p * noise

                decomposed_eps = self.vae.decompose_eps(eps)    # [[B, D1, 1, 1], [B, N*4, 1, 1]]
                vae_latent_point_dim = self.dae.num_classes
                output['vis/eps'] = decomposed_eps[1].view(
                    -1, self.dae.num_points, vae_latent_point_dim)[:, :, :3]    # [B, N, 3], viz of local latent
                p_loss_list = []
                for latent_id, eps in enumerate(decomposed_eps):
                    noise_p = torch.randn(size=eps.size(), device=device)   # [B, 128 or 8192, 1, 1], the new added noise.
                    eps_t_p = diffusion.sample_q(eps, noise_p, var_t_p, m_t_p)  # input in first epoch: [B, 128, 1, 1]*2, [B, 1, 1, 1]*2, output: [B, z(128) or N*4(8192), 1, 1]
                    # run the score model
                    eps_t_p.requires_grad_(True)
                    mixing_component = diffusion.mixing_component(  # None
                        eps_t_p, var_t_p, t_p, enabled=args.mixed_prediction)
                    if latent_id == 0:
                        pred_params_p = dae[latent_id]( # predicted noise
                            eps_t_p, t_p, x0=eps, clip_feat=clip_feat, **dae_kwarg) # [B, z(128), 1, 1]
                    else:
                        condition_input = decomposed_eps[0] if not self.cfg.data.cond_on_cat else \
                            torch.cat(
                                [decomposed_eps[0], output['cls_emb'].unsqueeze(-1).unsqueeze(-1)], dim=1)  # z0: [B, z(128), 1, 1]
                        condition_input = self.model.global2style(
                            condition_input)    # [B, z(128), 1, 1], a mlp here but not used.
                        pred_params_p = dae[latent_id](eps_t_p, t_p, x0=eps,
                                                       condition_input=condition_input, clip_feat=clip_feat, part_types=data['part_types'])    # [B, N*4(8192), 1, 1]

                    # pred_eps_t0 = (eps_t_p - torch.sqrt(var_t_p)    # recovered x0
                    #                * pred_params_p) / m_t_p

                    params = utils.get_mixed_prediction(args.mixed_prediction,  # [B, 128, 1, 1]
                                                        pred_params_p, dae[latent_id].mixing_logit, mixing_component)   # not used for z, pred_params_p == params
                    if self.cfg.latent_pts.pvd_mse_loss:    # here
                        if latent_id == 0:
                            p_loss = F.mse_loss(
                                params.contiguous().view(B, -1), noise_p.view(B, -1),
                                reduction='mean')
                        else:
                            # p_loss = torch.mean(torch.mean((params.contiguous().view(B, -1) - noise_p.view(B, -1)) ** 2, dim=-1) * data['train_weight'].view(B).to(params.device))
                            if not self.cfg.latent_pts.pred_part_type:
                                p_loss = torch.mean(torch.mean((params.contiguous().view(B, -1) - noise_p.view(B, -1)) ** 2, dim=-1))
                            else:
                                params = params.contiguous().view(B, -1).view(B, self.dae.num_points, -1)
                                params_h = params[:, :, :vae_latent_point_dim]
                                params_seg = params[data['seg_mask'], :, vae_latent_point_dim:]
                                noise_p = noise_p.contiguous().view(B, -1).view(B, self.dae.num_points, -1)
                                noise_h = noise_p[:, :, :vae_latent_point_dim]
                                noise_seg = noise_p[data['seg_mask'], :, vae_latent_point_dim:]
                                # if 'pred_seg_weights' in data.keys():
                                #     pred_seg_weights = data['pred_seg_weights'].to(params.device)
                                #     p_loss = (((params_h - noise_h) ** 2).mean(-1) * pred_seg_weights).mean()
                                # else:
                                p_loss = torch.mean((params_h - noise_h) ** 2)
                                # self.save_latent_pts(tr_pts, eps.reshape(B, -1, 4), data['part_types'], params_seg, '.')
                                if data['seg_mask'].any():
                                    if self.cfg.latent_pts.pred_part_type:
                                        # p_loss_seg = torch.mean((params_seg - data['part_types']) ** 2) * 2
                                        # p_loss_seg = F.cross_entropy(params_seg.view(-1, self.cfg.data.num_parts), data['part_types'].to(params_seg.device).argmax(-1).view(-1), reduction='mean')
                                        
                                        B_label = data['seg_mask'].sum().item()
                                        p_loss_seg = F.cross_entropy(params_seg.view(-1, self.cfg.data.num_parts), \
                                                    data['part_types'][data['seg_mask']].to(params_seg.device).argmax(-1).view(-1), reduction='none').reshape(B_label, -1)
                                        weight_seg = torch.ones_like(p_loss_seg).to(p_loss_seg.device)
                                        weight_seg *= ((1000 - t_p[data['seg_mask']]) / 1000).view(B_label, 1)  # FIXME
                                        # weight_seg[data['part_types'].to(params_seg.device).argmax(-1) != 3] *= 3 # car
                                        # weight_seg[data['part_types'].to(params_seg.device).argmax(-1) != 0] *= 10  # intra
                                        weight_seg *= data['pred_seg_weights'][data['seg_mask']].to(p_loss_seg.device)
                                        p_loss_seg = (p_loss_seg * weight_seg).mean()

                                        mIOU = params_seg.argmax(-1).eq(data['part_types'][data['seg_mask']].to(params_seg.device).argmax(-1)).float().mean(-1)   # [B]
                                        for miou, t in zip(mIOU, t_p[data['seg_mask']]):
                                            if t not in self.mIOU_list:
                                                self.mIOU_list[t.item()] = utils.AvgrageMeter()
                                            self.mIOU_list[t.item()].update(miou.item())
                    else:
                        l2_term_p = torch.square(params - noise_p)
                        p_objective = torch.sum(
                            obj_weight_t_p * l2_term_p, dim=[1, 2, 3])
                        regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff, \
                            jac_reg_loss, kin_reg_loss = utils.dae_regularization(
                                args, dae_sn_calculator, diffusion, dae, step, t_p,
                                pred_params_p, eps_t_p, var_t_p, m_t_p, g2_t_p)
                        reg_mlogit = ((torch.sum(torch.sigmoid(dae.mixing_logit)) -
                                       args.regularize_mlogit_margin)**2) * args.regularize_mlogit \
                            if args.regularize_mlogit else 0
                        p_loss = torch.mean(p_objective) + \
                            regularization_p + reg_mlogit
                    if self.writer is not None:
                        self.writer.avg_meter(
                            'train/p_loss_%d' % latent_id, p_loss.detach().item())
                        if (self.cfg.latent_pts.pred_part_type) and latent_id == 1 and data['seg_mask'].any():
                            self.writer.avg_meter(
                                'train/p_seg_loss_%d' % latent_id, p_loss_seg.detach().item())
                    p_loss_list.append(p_loss)
                if (self.cfg.latent_pts.pred_part_type) and data['seg_mask'].any():
                    p_loss_list.append(p_loss_seg)
            p_loss = sum(p_loss_list)  # torch.cat(p_loss_list, dim=0).sum()
            loss = p_loss
            # update dae parameters
            # p_loss.retain_grad()
            # params_h.retain_grad()
            # params_seg.retain_grad()
            grad_scalar.scale(p_loss).backward()
            utils.average_gradients(dae.parameters(), distributed)
            if args.grad_clip_max_norm > 0.:         # apply gradient clipping
                grad_scalar.unscale_(dae_optimizer)
                torch.nn.utils.clip_grad_norm_(dae.parameters(),
                                               max_norm=args.grad_clip_max_norm)
            grad_scalar.step(dae_optimizer)

            # update grade scalar
            grad_scalar.update()

            if args.bound_mlogit:
                dae.mixing_logit.data.clamp_(max=args.bound_mlogit_value)
            # Bookkeeping!
            writer = self.writer
            if writer is not None:
                writer.avg_meter('train/lr_dae', dae_optimizer.state_dict()[
                    'param_groups'][0]['lr'], global_step)
                writer.avg_meter('train/lr_vae', vae_optimizer.state_dict()[
                    'param_groups'][0]['lr'], global_step)
                if self.cfg.latent_pts.pvd_mse_loss:
                    writer.avg_meter(
                        'train/p_loss', p_loss.item(), global_step)
                    if args.mixed_prediction and global_step % 500 == 0:
                        for i in range(len(dae)):
                            m = torch.sigmoid(dae[i].mixing_logit)
                            if not torch.isnan(m).any():
                                writer.add_histogram(
                                    'mixing_prob_%d' % i, m.detach().cpu().numpy(), global_step)

                    # no other loss
                else:
                    writer.avg_meter(
                        'train/p_loss', (p_loss - regularization_p).item(), global_step)
                    if torch.is_tensor(regularization_p):
                        writer.avg_meter(
                            'train/reg_p', regularization_p.item(), global_step)
                    if args.regularize_mlogit:
                        writer.avg_meter(
                            'train/m_logit', reg_mlogit / args.regularize_mlogit, global_step)
                    if args.mixed_prediction:
                        writer.avg_meter(
                            'train/m_logit_sum', torch.sum(torch.sigmoid(dae.mixing_logit)).detach().cpu(), global_step)
                    if (global_step) % 500 == 0:
                        writer.add_scalar(
                            'train/norm_loss_dae', dae_norm_loss, global_step)
                        writer.add_scalar('train/bn_loss_dae',
                                          dae_bn_loss, global_step)
                        writer.add_scalar(
                            'train/norm_coeff_dae', dae_wdn_coeff, global_step)
                        if args.mixed_prediction:
                            m = torch.sigmoid(dae.mixing_logit)
                            if not torch.isnan(m).any():
                                writer.add_histogram(
                                    'mixing_prob', m.detach().cpu().numpy(), global_step)

        # write stats
        if self.writer is not None:
            for k, v in output.items():
                if 'print/' in k and step is not None:
                    self.writer.avg_meter(k.split('print/')[-1],
                                          v.mean().item() if torch.is_tensor(v) else v,
                                          step=step)
        res = output
        output_dict = {
            'loss': loss.detach().cpu().item(),
            'x_0_pred': res['x_0_pred'].detach().cpu(),  # perturbed data
            'x_0': res['x_0'].detach().cpu(),
            # B.B,3
            'x_t': res['final_pred'].detach().view(batch_size, -1, res['x_0'].shape[-1]),
            't': res.get('t', None)
        }

        for k, v in output.items():
            if 'vis/' in k:
                output_dict[k] = v
        return output_dict
    # --------------------------------------------- #
    #   visulization function and sampling function #
    # --------------------------------------------- #

    def build_prior(self):
        args = self.cfg.sde
        device = torch.device(self.device_str)
        arch_instance_dae = utils.get_arch_cells_denoising(
            'res_ho_attn', True, False)
        num_input_channels = self.cfg.shapelatent.latent_dim

        DAE = nn.ModuleList(
            [
                import_model(self.cfg.latent_pts.style_prior)(args,
                                                              self.cfg.latent_pts.style_dim, self.cfg),  # style prior
                import_model(self.cfg.sde.prior_model)(args,
                                                       num_input_channels, self.cfg),  # global prior, conditional model
            ])

        self.dae = DAE.to(device)

        # Bad solution! it is used in validate_inspect function
        self.dae.num_points = self.dae[1].num_points
        self.dae.num_classes = self.dae[1].num_classes

        if len(self.cfg.sde.dae_checkpoint):
            logger.info('Load dae checkpoint: {}',
                        self.cfg.sde.dae_checkpoint)
            checkpoint = torch.load(
                self.cfg.sde.dae_checkpoint, map_location='cpu')
            self.dae.load_state_dict(checkpoint['dae_state_dict'])

        self.diffusion_cont = make_diffusion(args)  # continuous?
        self.diffusion_disc = DiffusionDiscretized(
            args, self.diffusion_cont.var, self.cfg)
        if not quiet:
            logger.info('DAE: {}', self.dae)
        logger.info('DAE: param size = %fM ' %
                    utils.count_parameters_in_M(self.dae))
        # sync all parameters between all gpus by sending param from rank 0 to all gpus.
        utils.broadcast_params(self.dae.parameters(), self.args.distributed)
