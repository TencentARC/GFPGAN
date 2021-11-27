# general settings
name: train_GFPGANv1_512_simple
model_type: GFPGANModel
num_gpu: auto  # officially, we use 4 GPUs
manual_seed: 0

# dataset and data loader settings
datasets:
  train:
    name: FFHQ
    type: FFHQDegradationDataset
    # dataroot_gt: datasets/ffhq/ffhq_512.lmdb
    dataroot_gt: datasets/ffhq/ffhq_512
    io_backend:
      # type: lmdb
      type: disk

    use_hflip: true
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
    out_size: 512

    blur_kernel_size: 41
    kernel_list: ['iso', 'aniso']
    kernel_prob: [0.5, 0.5]
    blur_sigma: [0.1, 10]
    downsample_range: [0.8, 8]
    noise_range: [0, 20]
    jpeg_range: [60, 100]

    # color jitter and gray
    color_jitter_prob: 0.3
    color_jitter_shift: 20
    color_jitter_pt_prob: 0.3
    gray_prob: 0.01

    # If you do not want colorization, please set
    # color_jitter_prob: ~
    # color_jitter_pt_prob: ~
    # gray_prob: 0.01
    # gt_gray: True

    # data loader
    use_shuffle: true
    num_worker_per_gpu: 6
    batch_size_per_gpu: 3
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    # Please modify accordingly to use your own validation
    # Or comment the val block if do not need validation during training
    name: validation
    type: PairedImageDataset
    dataroot_lq: datasets/faces/validation/input
    dataroot_gt: datasets/faces/validation/reference
    io_backend:
      type: disk
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
    scale: 1

# network structures
network_g:
  type: GFPGANv1
  out_size: 512
  num_style_feat: 512
  channel_multiplier: 1
  resample_kernel: [1, 3, 3, 1]
  decoder_load_path: experiments/pretrained_models/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth
  fix_decoder: true
  num_mlp: 8
  lr_mlp: 0.01
  input_is_latent: true
  different_w: true
  narrow: 1
  sft_half: true

network_d:
  type: StyleGAN2Discriminator
  out_size: 512
  channel_multiplier: 1
  resample_kernel: [1, 3, 3, 1]


# path
path:
  pretrain_network_g: ~
  param_key_g: params_ema
  strict_load_g: ~
  pretrain_network_d: ~
  resume_state: ~

# training settings
train:
  optim_g:
    type: Adam
    lr: !!float 2e-3
  optim_d:
    type: Adam
    lr: !!float 2e-3
  optim_component:
    type: Adam
    lr: !!float 2e-3

  scheduler:
    type: MultiStepLR
    milestones: [600000, 700000]
    gamma: 0.5

  total_iter: 800000
  warmup_iter: -1  # no warm up

  # losses
  # pixel loss
  pixel_opt:
    type: L1Loss
    loss_weight: !!float 1e-1
    reduction: mean
  # L1 loss used in pyramid loss, component style loss and identity loss
  L1_opt:
    type: L1Loss
    loss_weight: 1
    reduction: mean

  # image pyramid loss
  pyramid_loss_weight: 1
  remove_pyramid_loss: 50000
  # perceptual loss (content and style losses)
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      # before relu
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1
      'conv4_4': 1
      'conv5_4': 1
    vgg_type: vgg19
    use_input_norm: true
    perceptual_weight: !!float 1
    style_weight: 50
    range_norm: true
    criterion: l1
  # gan loss
  gan_opt:
    type: GANLoss
    gan_type: wgan_softplus
    loss_weight: !!float 1e-1
  # r1 regularization for discriminator
  r1_reg_weight: 10

  net_d_iters: 1
  net_d_init_iters: 0
  net_d_reg_every: 16

# validation settings
val:
  val_freq: !!float 5e3
  save_img: true

  metrics:
    psnr: # metric name
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false

# logging settings
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# dist training settings
dist_params:
  backend: nccl
  port: 29500

find_unused_parameters: true
