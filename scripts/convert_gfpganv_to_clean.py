import argparse
import math
import torch

from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean


def modify_checkpoint(checkpoint_bilinear, checkpoint_clean):
    for ori_k, ori_v in checkpoint_bilinear.items():
        if 'stylegan_decoder' in ori_k:
            if 'style_mlp' in ori_k:  # style_mlp_layers
                lr_mul = 0.01
                prefix, name, idx, var = ori_k.split('.')
                idx = (int(idx) * 2) - 1
                crt_k = f'{prefix}.{name}.{idx}.{var}'
                if var == 'weight':
                    _, c_in = ori_v.size()
                    scale = (1 / math.sqrt(c_in)) * lr_mul
                    crt_v = ori_v * scale * 2**0.5
                else:
                    crt_v = ori_v * lr_mul * 2**0.5
                checkpoint_clean[crt_k] = crt_v
            elif 'modulation' in ori_k:  # modulation in StyleConv
                lr_mul = 1
                crt_k = ori_k
                var = ori_k.split('.')[-1]
                if var == 'weight':
                    _, c_in = ori_v.size()
                    scale = (1 / math.sqrt(c_in)) * lr_mul
                    crt_v = ori_v * scale
                else:
                    crt_v = ori_v * lr_mul
                checkpoint_clean[crt_k] = crt_v
            elif 'style_conv' in ori_k:
                # StyleConv in style_conv1 and style_convs
                if 'activate' in ori_k:  # FusedLeakyReLU
                    # eg. style_conv1.activate.bias
                    # eg. style_convs.13.activate.bias
                    split_rlt = ori_k.split('.')
                    if len(split_rlt) == 4:
                        prefix, name, _, var = split_rlt
                        crt_k = f'{prefix}.{name}.{var}'
                    elif len(split_rlt) == 5:
                        prefix, name, idx, _, var = split_rlt
                        crt_k = f'{prefix}.{name}.{idx}.{var}'
                    crt_v = ori_v * 2**0.5  # 2**0.5 used in FusedLeakyReLU
                    c = crt_v.size(0)
                    checkpoint_clean[crt_k] = crt_v.view(1, c, 1, 1)
                elif 'modulated_conv' in ori_k:
                    # eg. style_conv1.modulated_conv.weight
                    # eg. style_convs.13.modulated_conv.weight
                    _, c_out, c_in, k1, k2 = ori_v.size()
                    scale = 1 / math.sqrt(c_in * k1 * k2)
                    crt_k = ori_k
                    checkpoint_clean[crt_k] = ori_v * scale
                elif 'weight' in ori_k:
                    crt_k = ori_k
                    checkpoint_clean[crt_k] = ori_v * 2**0.5
            elif 'to_rgb' in ori_k:  # StyleConv in to_rgb1 and to_rgbs
                if 'modulated_conv' in ori_k:
                    # eg. to_rgb1.modulated_conv.weight
                    # eg. to_rgbs.5.modulated_conv.weight
                    _, c_out, c_in, k1, k2 = ori_v.size()
                    scale = 1 / math.sqrt(c_in * k1 * k2)
                    crt_k = ori_k
                    checkpoint_clean[crt_k] = ori_v * scale
                else:
                    crt_k = ori_k
                    checkpoint_clean[crt_k] = ori_v
            else:
                crt_k = ori_k
                checkpoint_clean[crt_k] = ori_v
            # end of 'stylegan_decoder'
        elif 'conv_body_first' in ori_k or 'final_conv' in ori_k:
            # key name
            name, _, var = ori_k.split('.')
            crt_k = f'{name}.{var}'
            # weight and bias
            if var == 'weight':
                c_out, c_in, k1, k2 = ori_v.size()
                scale = 1 / math.sqrt(c_in * k1 * k2)
                checkpoint_clean[crt_k] = ori_v * scale * 2**0.5
            else:
                checkpoint_clean[crt_k] = ori_v * 2**0.5
        elif 'conv_body' in ori_k:
            if 'conv_body_up' in ori_k:
                ori_k = ori_k.replace('conv2.weight', 'conv2.1.weight')
                ori_k = ori_k.replace('skip.weight', 'skip.1.weight')
            name1, idx1, name2, _, var = ori_k.split('.')
            crt_k = f'{name1}.{idx1}.{name2}.{var}'
            if name2 == 'skip':
                c_out, c_in, k1, k2 = ori_v.size()
                scale = 1 / math.sqrt(c_in * k1 * k2)
                checkpoint_clean[crt_k] = ori_v * scale / 2**0.5
            else:
                if var == 'weight':
                    c_out, c_in, k1, k2 = ori_v.size()
                    scale = 1 / math.sqrt(c_in * k1 * k2)
                    checkpoint_clean[crt_k] = ori_v * scale
                else:
                    checkpoint_clean[crt_k] = ori_v
                if 'conv1' in ori_k:
                    checkpoint_clean[crt_k] *= 2**0.5
        elif 'toRGB' in ori_k:
            crt_k = ori_k
            if 'weight' in ori_k:
                c_out, c_in, k1, k2 = ori_v.size()
                scale = 1 / math.sqrt(c_in * k1 * k2)
                checkpoint_clean[crt_k] = ori_v * scale
            else:
                checkpoint_clean[crt_k] = ori_v
        elif 'final_linear' in ori_k:
            crt_k = ori_k
            if 'weight' in ori_k:
                _, c_in = ori_v.size()
                scale = 1 / math.sqrt(c_in)
                checkpoint_clean[crt_k] = ori_v * scale
            else:
                checkpoint_clean[crt_k] = ori_v
        elif 'condition' in ori_k:
            crt_k = ori_k
            if '0.weight' in ori_k:
                c_out, c_in, k1, k2 = ori_v.size()
                scale = 1 / math.sqrt(c_in * k1 * k2)
                checkpoint_clean[crt_k] = ori_v * scale * 2**0.5
            elif '0.bias' in ori_k:
                checkpoint_clean[crt_k] = ori_v * 2**0.5
            elif '2.weight' in ori_k:
                c_out, c_in, k1, k2 = ori_v.size()
                scale = 1 / math.sqrt(c_in * k1 * k2)
                checkpoint_clean[crt_k] = ori_v * scale
            elif '2.bias' in ori_k:
                checkpoint_clean[crt_k] = ori_v

    return checkpoint_clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, help='Path to the original model')
    parser.add_argument('--narrow', type=float, default=1)
    parser.add_argument('--channel_multiplier', type=float, default=2)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    ori_ckpt = torch.load(args.ori_path)['params_ema']

    net = GFPGANv1Clean(
        512,
        num_style_feat=512,
        channel_multiplier=args.channel_multiplier,
        decoder_load_path=None,
        fix_decoder=False,
        # for stylegan decoder
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=args.narrow,
        sft_half=True)
    crt_ckpt = net.state_dict()

    crt_ckpt = modify_checkpoint(ori_ckpt, crt_ckpt)
    print(f'Save to {args.save_path}.')
    torch.save(dict(params_ema=crt_ckpt), args.save_path, _use_new_zipfile_serialization=False)
