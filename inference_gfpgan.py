import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--arch', type=str, default='clean')
    parser.add_argument('--channel', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan')
    parser.add_argument('--bg_tile', type=int, default=0)
    parser.add_argument('--test_path', type=str, default='inputs/whole_imgs')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--paste_back', action='store_false')
    parser.add_argument('--save_root', type=str, default='results')

    args = parser.parse_args()
    if args.test_path.endswith('/'):
        args.test_path = args.test_path[:-1]
    os.makedirs(args.save_root, exist_ok=True)

    # background upsampler
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from realesrgan import RealESRGANer
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None
    # set up GFPGAN restorer
    restorer = GFPGANer(
        model_path=args.model_path,
        upscale=args.upscale,
        arch=args.arch,
        channel_multiplier=args.channel,
        bg_upsampler=bg_upsampler)

    img_list = sorted(glob.glob(os.path.join(args.test_path, '*')))
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=args.paste_back)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(args.save_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(restored_face, save_crop_path)
            # save restored face
            if args.suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.save_root, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save cmp image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.save_root, 'cmp', f'{basename}_{idx:02d}.png'))
        # save restored img
        if args.suffix is not None:
            save_restore_path = os.path.join(args.save_root, 'restored_imgs', f'{basename}_{args.suffix}{ext}')
        else:
            save_restore_path = os.path.join(args.save_root, 'restored_imgs', img_name)
        imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{args.save_root}] folder.')


if __name__ == '__main__':
    main()
