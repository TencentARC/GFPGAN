import torch

from gfpgan.archs.stylegan2_clean_arch import StyleGAN2GeneratorClean


def test_stylegan2generatorclean():
    """Test arch: StyleGAN2GeneratorClean."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = StyleGAN2GeneratorClean(
            out_size=32, num_style_feat=512, num_mlp=8, channel_multiplier=1, narrow=0.5).cuda().eval()
        style = torch.rand((1, 512), dtype=torch.float32).cuda()
        output = net([style], input_is_latent=False)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # -------------------- with return_latents ----------------------- #
        output = net([style], input_is_latent=True, return_latents=True)
        assert output[0].shape == (1, 3, 32, 32)
        assert len(output[1]) == 1
        # check latent
        assert output[1][0].shape == (8, 512)

        # -------------------- with randomize_noise = False ----------------------- #
        output = net([style], randomize_noise=False)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # -------------------- with truncation = 0.5 and mixing----------------------- #
        output = net([style, style], truncation=0.5, truncation_latent=style)
        assert output[0].shape == (1, 3, 32, 32)
        assert output[1] is None

        # ------------------ test make_noise ----------------------- #
        out = net.make_noise()
        assert len(out) == 7
        assert out[0].shape == (1, 1, 4, 4)
        assert out[1].shape == (1, 1, 8, 8)
        assert out[2].shape == (1, 1, 8, 8)
        assert out[3].shape == (1, 1, 16, 16)
        assert out[4].shape == (1, 1, 16, 16)
        assert out[5].shape == (1, 1, 32, 32)
        assert out[6].shape == (1, 1, 32, 32)

        # ------------------ test get_latent ----------------------- #
        out = net.get_latent(style)
        assert out.shape == (1, 512)

        # ------------------ test mean_latent ----------------------- #
        out = net.mean_latent(2)
        assert out.shape == (1, 512)
