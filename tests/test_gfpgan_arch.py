import torch

from gfpgan.archs.gfpganv1_arch import FacialComponentDiscriminator, GFPGANv1, StyleGAN2GeneratorSFT
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean, StyleGAN2GeneratorCSFT


def test_stylegan2generatorsft():
    """Test arch: StyleGAN2GeneratorSFT."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = StyleGAN2GeneratorSFT(
            out_size=32,
            num_style_feat=512,
            num_mlp=8,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            lr_mlp=0.01,
            narrow=1,
            sft_half=False).cuda().eval()
        style = torch.rand((1, 512), dtype=torch.float32).cuda()
        condition1 = torch.rand((1, 512, 8, 8), dtype=torch.float32).cuda()
        condition2 = torch.rand((1, 512, 16, 16), dtype=torch.float32).cuda()
        condition3 = torch.rand((1, 512, 32, 32), dtype=torch.float32).cuda()
        conditions = [condition1, condition1, condition2, condition2, condition3, condition3]
        output = net([style], conditions)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # -------------------- with return_latents ----------------------- #
        output = net([style], conditions, return_latents=True)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 1
        # check latent
        assert output[1][0].shape == (8, 512)

        # -------------------- with randomize_noise = False ----------------------- #
        output = net([style], conditions, randomize_noise=False)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # -------------------- with truncation = 0.5 and mixing----------------------- #
        output = net([style, style], conditions, truncation=0.5, truncation_latent=style)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None


def test_gfpganv1():
    """Test arch: GFPGANv1."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = GFPGANv1(
            out_size=32,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=True).cuda().eval()
        img = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
        output = net(img)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 3
        # check out_rgbs for intermediate loss
        assert output[1][0].shape == (1, 3, 8, 8)
        assert output[1][1].shape == (1, 3, 16, 16)
        assert output[1][2].shape == (1, 3, 32, 32)

        # -------------------- with different_w = True ----------------------- #
        net = GFPGANv1(
            out_size=32,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=True,
            narrow=1,
            sft_half=True).cuda().eval()
        img = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
        output = net(img)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 3
        # check out_rgbs for intermediate loss
        assert output[1][0].shape == (1, 3, 8, 8)
        assert output[1][1].shape == (1, 3, 16, 16)
        assert output[1][2].shape == (1, 3, 32, 32)


def test_facialcomponentdiscriminator():
    """Test arch: FacialComponentDiscriminator."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = FacialComponentDiscriminator().cuda().eval()
        img = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
        output = net(img)
        assert len(output) == 2
        assert output[0].shape == (1, 1, 8, 8)
        assert output[1] is None

        # -------------------- return intermediate features ----------------------- #
        output = net(img, return_feats=True)
        assert len(output) == 2
        assert output[0].shape == (1, 1, 8, 8)
        assert len(output[1]) == 2
        assert output[1][0].shape == (1, 128, 16, 16)
        assert output[1][1].shape == (1, 256, 8, 8)


def test_stylegan2generatorcsft():
    """Test arch: StyleGAN2GeneratorCSFT."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = StyleGAN2GeneratorCSFT(
            out_size=32, num_style_feat=512, num_mlp=8, channel_multiplier=1, narrow=1, sft_half=False).cuda().eval()
        style = torch.rand((1, 512), dtype=torch.float32).cuda()
        condition1 = torch.rand((1, 512, 8, 8), dtype=torch.float32).cuda()
        condition2 = torch.rand((1, 512, 16, 16), dtype=torch.float32).cuda()
        condition3 = torch.rand((1, 512, 32, 32), dtype=torch.float32).cuda()
        conditions = [condition1, condition1, condition2, condition2, condition3, condition3]
        output = net([style], conditions)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # -------------------- with return_latents ----------------------- #
        output = net([style], conditions, return_latents=True)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 1
        # check latent
        assert output[1][0].shape == (8, 512)

        # -------------------- with randomize_noise = False ----------------------- #
        output = net([style], conditions, randomize_noise=False)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # -------------------- with truncation = 0.5 and mixing----------------------- #
        output = net([style, style], conditions, truncation=0.5, truncation_latent=style)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None


def test_gfpganv1clean():
    """Test arch: GFPGANv1Clean."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = GFPGANv1Clean(
            out_size=32,
            num_style_feat=512,
            channel_multiplier=1,
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=True).cuda().eval()

        img = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
        output = net(img)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 3
        # check out_rgbs for intermediate loss
        assert output[1][0].shape == (1, 3, 8, 8)
        assert output[1][1].shape == (1, 3, 16, 16)
        assert output[1][2].shape == (1, 3, 32, 32)

        # -------------------- with different_w = True ----------------------- #
        net = GFPGANv1Clean(
            out_size=32,
            num_style_feat=512,
            channel_multiplier=1,
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            input_is_latent=False,
            different_w=True,
            narrow=1,
            sft_half=True).cuda().eval()
        img = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
        output = net(img)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 3
        # check out_rgbs for intermediate loss
        assert output[1][0].shape == (1, 3, 8, 8)
        assert output[1][1].shape == (1, 3, 16, 16)
        assert output[1][2].shape == (1, 3, 32, 32)
