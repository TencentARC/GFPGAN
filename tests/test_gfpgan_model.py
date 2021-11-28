import tempfile
import torch
import yaml
from basicsr.archs.stylegan2_arch import StyleGAN2Discriminator
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.losses.losses import GANLoss, L1Loss, PerceptualLoss

from gfpgan.archs.arcface_arch import ResNetArcFace
from gfpgan.archs.gfpganv1_arch import FacialComponentDiscriminator, GFPGANv1
from gfpgan.models.gfpgan_model import GFPGANModel


def test_gfpgan_model():
    with open('tests/data/test_gfpgan_model.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    # build model
    model = GFPGANModel(opt)
    # test attributes
    assert model.__class__.__name__ == 'GFPGANModel'
    assert isinstance(model.net_g, GFPGANv1)  # generator
    assert isinstance(model.net_d, StyleGAN2Discriminator)  # discriminator
    # facial component discriminators
    assert isinstance(model.net_d_left_eye, FacialComponentDiscriminator)
    assert isinstance(model.net_d_right_eye, FacialComponentDiscriminator)
    assert isinstance(model.net_d_mouth, FacialComponentDiscriminator)
    # identity network
    assert isinstance(model.network_identity, ResNetArcFace)
    # losses
    assert isinstance(model.cri_pix, L1Loss)
    assert isinstance(model.cri_perceptual, PerceptualLoss)
    assert isinstance(model.cri_gan, GANLoss)
    assert isinstance(model.cri_l1, L1Loss)
    # optimizer
    assert isinstance(model.optimizers[0], torch.optim.Adam)
    assert isinstance(model.optimizers[1], torch.optim.Adam)

    # prepare data
    gt = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    lq = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    loc_left_eye = torch.rand((1, 4), dtype=torch.float32)
    loc_right_eye = torch.rand((1, 4), dtype=torch.float32)
    loc_mouth = torch.rand((1, 4), dtype=torch.float32)
    data = dict(gt=gt, lq=lq, loc_left_eye=loc_left_eye, loc_right_eye=loc_right_eye, loc_mouth=loc_mouth)
    model.feed_data(data)
    # check data shape
    assert model.lq.shape == (1, 3, 512, 512)
    assert model.gt.shape == (1, 3, 512, 512)
    assert model.loc_left_eyes.shape == (1, 4)
    assert model.loc_right_eyes.shape == (1, 4)
    assert model.loc_mouths.shape == (1, 4)

    # ----------------- test optimize_parameters -------------------- #
    model.feed_data(data)
    model.optimize_parameters(1)
    assert model.output.shape == (1, 3, 512, 512)
    assert isinstance(model.log_dict, dict)
    # check returned keys
    expected_keys = [
        'l_g_pix', 'l_g_percep', 'l_g_style', 'l_g_gan', 'l_g_gan_left_eye', 'l_g_gan_right_eye', 'l_g_gan_mouth',
        'l_g_comp_style_loss', 'l_identity', 'l_d', 'real_score', 'fake_score', 'l_d_r1', 'l_d_left_eye',
        'l_d_right_eye', 'l_d_mouth'
    ]
    assert set(expected_keys).issubset(set(model.log_dict.keys()))

    # ----------------- remove pyramid_loss_weight-------------------- #
    model.feed_data(data)
    model.optimize_parameters(100000)  # large than remove_pyramid_loss = 50000
    assert model.output.shape == (1, 3, 512, 512)
    assert isinstance(model.log_dict, dict)
    # check returned keys
    expected_keys = [
        'l_g_pix', 'l_g_percep', 'l_g_style', 'l_g_gan', 'l_g_gan_left_eye', 'l_g_gan_right_eye', 'l_g_gan_mouth',
        'l_g_comp_style_loss', 'l_identity', 'l_d', 'real_score', 'fake_score', 'l_d_r1', 'l_d_left_eye',
        'l_d_right_eye', 'l_d_mouth'
    ]
    assert set(expected_keys).issubset(set(model.log_dict.keys()))

    # ----------------- test save -------------------- #
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['path']['models'] = tmpdir
        model.opt['path']['training_states'] = tmpdir
        model.save(0, 1)

    # ----------------- test the test function -------------------- #
    model.test()
    assert model.output.shape == (1, 3, 512, 512)
    # delete net_g_ema
    model.__delattr__('net_g_ema')
    model.test()
    assert model.output.shape == (1, 3, 512, 512)
    assert model.net_g.training is True  # should back to training mode after testing

    # ----------------- test nondist_validation -------------------- #
    # construct dataloader
    dataset_opt = dict(
        name='Demo',
        dataroot_gt='tests/data/gt',
        dataroot_lq='tests/data/gt',
        io_backend=dict(type='disk'),
        scale=4,
        phase='val')
    dataset = PairedImageDataset(dataset_opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    assert model.is_train is True
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['path']['visualization'] = tmpdir
        model.nondist_validation(dataloader, 1, None, save_img=True)
        assert model.is_train is True
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)

    # validation
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['is_train'] = False
        model.opt['val']['suffix'] = 'test'
        model.opt['path']['visualization'] = tmpdir
        model.opt['val']['pbar'] = True
        model.nondist_validation(dataloader, 1, None, save_img=True)
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)

        # if opt['val']['suffix'] is None
        model.opt['val']['suffix'] = None
        model.opt['name'] = 'demo'
        model.opt['path']['visualization'] = tmpdir
        model.nondist_validation(dataloader, 1, None, save_img=True)
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)
