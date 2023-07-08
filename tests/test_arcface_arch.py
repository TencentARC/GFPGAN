import torch

from gfpgan.archs.arcface_arch import BasicBlock, Bottleneck, ResNetArcFace


def test_resnetarcface():
    """Test arch: ResNetArcFace."""

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=True).cuda().eval()
        img = torch.rand((1, 1, 128, 128), dtype=torch.float32).cuda()
        output = net(img)
        assert output.shape == (1, 512)

        # -------------------- without SE block ----------------------- #
        net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=False).cuda().eval()
        output = net(img)
        assert output.shape == (1, 512)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=True).to(torch.device("mps")).eval()
        img = torch.rand((1, 1, 128, 128), dtype=torch.float32).to(torch.device("mps"))
        output = net(img)
        assert output.shape == (1, 512)

        # -------------------- without SE block ----------------------- #
        net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=False).to(torch.device("mps")).eval()
        output = net(img)
        assert output.shape == (1, 512)


def test_basicblock():
    if torch.cuda.is_available():
        """Test the BasicBlock in arcface_arch"""
        block = BasicBlock(1, 3, stride=1, downsample=None).cuda()
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32).cuda()
        output = block(img)
        assert output.shape == (1, 3, 12, 12)

        # ----------------- use the downsmaple module--------------- #
        downsample = torch.nn.UpsamplingNearest2d(scale_factor=0.5).cuda()
        block = BasicBlock(1, 3, stride=2, downsample=downsample).cuda()
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32).cuda()
        output = block(img)
        assert output.shape == (1, 3, 6, 6)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        """Test the BasicBlock in arcface_arch"""
        block = BasicBlock(1, 3, stride=1, downsample=None).to(torch.device("mps"))
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32, device=torch.device("mps"))
        output = block(img)
        assert output.shape == (1, 3, 12, 12)

        # ----------------- use the downsmaple module--------------- #
        downsample = torch.nn.UpsamplingNearest2d(scale_factor=0.5).to(torch.device("mps"))
        block = BasicBlock(1, 3, stride=2, downsample=downsample).to(torch.device("mps"))
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32, device=torch.device("mps"))
        output = block(img)
        assert output.shape == (1, 3, 6, 6)


def test_bottleneck():
    if torch.cuda.is_available():
        """Test the Bottleneck in arcface_arch"""
        block = Bottleneck(1, 1, stride=1, downsample=None).cuda()
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32).cuda()
        output = block(img)
        assert output.shape == (1, 4, 12, 12)

        # ----------------- use the downsmaple module--------------- #
        downsample = torch.nn.UpsamplingNearest2d(scale_factor=0.5).cuda()
        block = Bottleneck(1, 1, stride=2, downsample=downsample).cuda()
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32).cuda()
        output = block(img)
        assert output.shape == (1, 4, 6, 6)

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        """Test the Bottleneck in arcface_arch"""
        block = Bottleneck(1, 1, stride=1, downsample=None).to(torch.device("mps"))
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32, device=torch.device("mps"))
        output = block(img)
        assert output.shape == (1, 4, 12, 12)

        # ----------------- use the downsmaple module--------------- #
        downsample = torch.nn.UpsamplingNearest2d(scale_factor=0.5).to(torch.device("mps"))
        block = Bottleneck(1, 1, stride=2, downsample=downsample).to(torch.device("mps"))
        img = torch.rand((1, 1, 12, 12), dtype=torch.float32, device=torch.device("mps"))
        output = block(img)
        assert output.shape == (1, 4, 6, 6)
