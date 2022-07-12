import math
import os.path as osp
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.gan_loss import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm


@MODEL_REGISTRY.register()
class GFPGANModel(BaseModel):
    """The GFPGAN model for Towards real-world blind face restoratin with generative facial prior"""

    def __init__(self, opt):
        super(GFPGANModel, self).__init__(opt)
        self.idx = 0  # it is used for saving data for check

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.log_size = int(math.log(self.opt['network_g']['out_size'], 2))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        # ----------- define net_d ----------- #
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        # ----------- define net_g with Exponential Moving Average (EMA) ----------- #
        # net_g_ema only used for testing on one GPU and saving. There is no need to wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_d.train()
        self.net_g_ema.eval()

        # ----------- facial component networks ----------- #
        if ('network_d_left_eye' in self.opt and 'network_d_right_eye' in self.opt and 'network_d_mouth' in self.opt):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        if self.use_facial_disc:
            # left eye
            self.net_d_left_eye = build_network(self.opt['network_d_left_eye'])
            self.net_d_left_eye = self.model_to_device(self.net_d_left_eye)
            self.print_network(self.net_d_left_eye)
            load_path = self.opt['path'].get('pretrain_network_d_left_eye')
            if load_path is not None:
                self.load_network(self.net_d_left_eye, load_path, True, 'params')
            # right eye
            self.net_d_right_eye = build_network(self.opt['network_d_right_eye'])
            self.net_d_right_eye = self.model_to_device(self.net_d_right_eye)
            self.print_network(self.net_d_right_eye)
            load_path = self.opt['path'].get('pretrain_network_d_right_eye')
            if load_path is not None:
                self.load_network(self.net_d_right_eye, load_path, True, 'params')
            # mouth
            self.net_d_mouth = build_network(self.opt['network_d_mouth'])
            self.net_d_mouth = self.model_to_device(self.net_d_mouth)
            self.print_network(self.net_d_mouth)
            load_path = self.opt['path'].get('pretrain_network_d_mouth')
            if load_path is not None:
                self.load_network(self.net_d_mouth, load_path, True, 'params')

            self.net_d_left_eye.train()
            self.net_d_right_eye.train()
            self.net_d_mouth.train()

            # ----------- define facial component gan loss ----------- #
            self.cri_component = build_loss(train_opt['gan_component_opt']).to(self.device)

        # ----------- define losses ----------- #
        # pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # L1 loss is used in pyramid loss, component style loss and identity loss
        self.cri_l1 = build_loss(train_opt['L1_opt']).to(self.device)

        # gan loss (wgan)
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # ----------- define identity loss ----------- #
        if 'network_identity' in self.opt:
            self.use_identity = True
        else:
            self.use_identity = False

        if self.use_identity:
            # define identity network
            self.network_identity = build_network(self.opt['network_identity'])
            self.network_identity = self.model_to_device(self.network_identity)
            self.print_network(self.network_identity)
            load_path = self.opt['path'].get('pretrain_network_identity')
            if load_path is not None:
                self.load_network(self.network_identity, load_path, True, None)
            self.network_identity.eval()
            for param in self.network_identity.parameters():
                param.requires_grad = False

        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every = train_opt['net_d_reg_every']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_g.named_parameters():
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # ----------- optimizer d ----------- #
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        normal_params = []
        for _, param in self.net_d.named_parameters():
            normal_params.append(param)
        optim_params_d = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_d']['lr']
        }]
        optim_type = train_opt['optim_d'].pop('type')
        lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
        betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
        self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
        self.optimizers.append(self.optimizer_d)

        # ----------- optimizers for facial component networks ----------- #
        if self.use_facial_disc:
            # setup optimizers for facial component discriminators
            optim_type = train_opt['optim_component'].pop('type')
            lr = train_opt['optim_component']['lr']
            # left eye
            self.optimizer_d_left_eye = self.get_optimizer(
                optim_type, self.net_d_left_eye.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_left_eye)
            # right eye
            self.optimizer_d_right_eye = self.get_optimizer(
                optim_type, self.net_d_right_eye.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_right_eye)
            # mouth
            self.optimizer_d_mouth = self.get_optimizer(
                optim_type, self.net_d_mouth.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_mouth)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'loc_left_eye' in data:
            # get facial component locations, shape (batch, 4)
            self.loc_left_eyes = data['loc_left_eye']
            self.loc_right_eyes = data['loc_right_eye']
            self.loc_mouths = data['loc_mouth']

        # uncomment to check data
        # import torchvision
        # if self.opt['rank'] == 0:
        #     import os
        #     os.makedirs('tmp/gt', exist_ok=True)
        #     os.makedirs('tmp/lq', exist_ok=True)
        #     print(self.idx)
        #     torchvision.utils.save_image(
        #         self.gt, f'tmp/gt/gt_{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
        #     torchvision.utils.save_image(
        #         self.lq, f'tmp/lq/lq{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
        #     self.idx = self.idx + 1

    def construct_img_pyramid(self):
        """Construct image pyramid for intermediate restoration loss"""
        pyramid_gt = [self.gt]
        down_img = self.gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        face_ratio = int(self.opt['network_g']['out_size'] / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        # output
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        # do not update facial component net_d
        if self.use_facial_disc:
            for p in self.net_d_left_eye.parameters():
                p.requires_grad = False
            for p in self.net_d_right_eye.parameters():
                p.requires_grad = False
            for p in self.net_d_mouth.parameters():
                p.requires_grad = False

        # image pyramid loss weight
        pyramid_loss_weight = self.opt['train'].get('pyramid_loss_weight', 0)
        if pyramid_loss_weight > 0 and current_iter > self.opt['train'].get('remove_pyramid_loss', float('inf')):
            pyramid_loss_weight = 1e-12  # very small weight to avoid unused param error
        if pyramid_loss_weight > 0:
            self.output, out_rgbs = self.net_g(self.lq, return_rgb=True)
            pyramid_gt = self.construct_img_pyramid()
        else:
            self.output, out_rgbs = self.net_g(self.lq, return_rgb=False)

        # get roi-align regions
        if self.use_facial_disc:
            self.get_roi_regions(eye_out_size=80, mouth_out_size=120)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # image pyramid loss
            if pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.cri_l1(out_rgbs[i], pyramid_gt[i]) * pyramid_loss_weight
                    l_g_total += l_pyramid
                    loss_dict[f'l_p_{2**(i+3)}'] = l_pyramid

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # facial component loss
            if self.use_facial_disc:
                # left eye
                fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(self.left_eyes, return_feats=True)
                l_g_gan = self.cri_component(fake_left_eye, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_left_eye'] = l_g_gan
                # right eye
                fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(self.right_eyes, return_feats=True)
                l_g_gan = self.cri_component(fake_right_eye, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_right_eye'] = l_g_gan
                # mouth
                fake_mouth, fake_mouth_feats = self.net_d_mouth(self.mouths, return_feats=True)
                l_g_gan = self.cri_component(fake_mouth, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_mouth'] = l_g_gan

                if self.opt['train'].get('comp_style_weight', 0) > 0:
                    # get gt feat
                    _, real_left_eye_feats = self.net_d_left_eye(self.left_eyes_gt, return_feats=True)
                    _, real_right_eye_feats = self.net_d_right_eye(self.right_eyes_gt, return_feats=True)
                    _, real_mouth_feats = self.net_d_mouth(self.mouths_gt, return_feats=True)

                    def _comp_style(feat, feat_gt, criterion):
                        return criterion(self._gram_mat(feat[0]), self._gram_mat(
                            feat_gt[0].detach())) * 0.5 + criterion(
                                self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

                    # facial component style loss
                    comp_style_loss = 0
                    comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
                    comp_style_loss = comp_style_loss * self.opt['train']['comp_style_weight']
                    l_g_total += comp_style_loss
                    loss_dict['l_g_comp_style_loss'] = comp_style_loss

            # identity loss
            if self.use_identity:
                identity_weight = self.opt['train']['identity_weight']
                # get gray images and resize
                out_gray = self.gray_resize_for_identity(self.output)
                gt_gray = self.gray_resize_for_identity(self.gt)

                identity_gt = self.network_identity(gt_gray).detach()
                identity_out = self.network_identity(out_gray)
                l_identity = self.cri_l1(identity_out, identity_gt) * identity_weight
                l_g_total += l_identity
                loss_dict['l_identity'] = l_identity

            l_g_total.backward()
            self.optimizer_g.step()

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

        # ----------- optimize net_d ----------- #
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()
        if self.use_facial_disc:
            for p in self.net_d_left_eye.parameters():
                p.requires_grad = True
            for p in self.net_d_right_eye.parameters():
                p.requires_grad = True
            for p in self.net_d_mouth.parameters():
                p.requires_grad = True
            self.optimizer_d_left_eye.zero_grad()
            self.optimizer_d_right_eye.zero_grad()
            self.optimizer_d_mouth.zero_grad()

        fake_d_pred = self.net_d(self.output.detach())
        real_d_pred = self.net_d(self.gt)
        l_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d'] = l_d
        # In WGAN, real_score should be positive and fake_score should be negative
        loss_dict['real_score'] = real_d_pred.detach().mean()
        loss_dict['fake_score'] = fake_d_pred.detach().mean()
        l_d.backward()

        # regularization loss
        if current_iter % self.net_d_reg_every == 0:
            self.gt.requires_grad = True
            real_pred = self.net_d(self.gt)
            l_d_r1 = r1_penalty(real_pred, self.gt)
            l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
            loss_dict['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()

        self.optimizer_d.step()

        # optimize facial component discriminators
        if self.use_facial_disc:
            # left eye
            fake_d_pred, _ = self.net_d_left_eye(self.left_eyes.detach())
            real_d_pred, _ = self.net_d_left_eye(self.left_eyes_gt)
            l_d_left_eye = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye
            l_d_left_eye.backward()
            # right eye
            fake_d_pred, _ = self.net_d_right_eye(self.right_eyes.detach())
            real_d_pred, _ = self.net_d_right_eye(self.right_eyes_gt)
            l_d_right_eye = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye
            l_d_right_eye.backward()
            # mouth
            fake_d_pred, _ = self.net_d_mouth(self.mouths.detach())
            real_d_pred, _ = self.net_d_mouth(self.mouths_gt)
            l_d_mouth = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth
            l_d_mouth.backward()

            self.optimizer_d_left_eye.step()
            self.optimizer_d_right_eye.step()
            self.optimizer_d_mouth.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output, _ = self.net_g_ema(self.lq)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output, _ = self.net_g(self.lq)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output.detach().cpu(), min_max=(-1, 1))
            metric_data['img'] = sr_img
            if hasattr(self, 'gt'):
                gt_img = tensor2img(self.gt.detach().cpu(), min_max=(-1, 1))
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        # save net_g and net_d
        self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        self.save_network(self.net_d, 'net_d', current_iter)
        # save component discriminators
        if self.use_facial_disc:
            self.save_network(self.net_d_left_eye, 'net_d_left_eye', current_iter)
            self.save_network(self.net_d_right_eye, 'net_d_right_eye', current_iter)
            self.save_network(self.net_d_mouth, 'net_d_mouth', current_iter)
        # save training state
        self.save_training_state(epoch, current_iter)
