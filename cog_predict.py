# flake8: noqa
# This file is used for deploying replicate models
# running: cog predict -i img=@inputs/whole_imgs/10045.png -i version='v1.4' -i scale=2
# push: cog push r8.im/tencentarc/gfpgan
# push (backup): cog push r8.im/xinntao/gfpgan

import os

os.system('python setup.py develop')
os.system('pip install realesrgan')

import cv2
import shutil
import tempfile
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact

from gfpgan import GFPGANer

try:
    from cog import BasePredictor, Input, Path
    from realesrgan.utils import RealESRGANer
except Exception:
    print('please install cog and realesrgan package')


class Predictor(BasePredictor):

    def setup(self):
        os.makedirs('output', exist_ok=True)
        # download weights
        if not os.path.exists('gfpgan/weights/realesr-general-x4v3.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./gfpgan/weights'
            )
        if not os.path.exists('gfpgan/weights/GFPGANv1.2.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P ./gfpgan/weights')
        if not os.path.exists('gfpgan/weights/GFPGANv1.3.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P ./gfpgan/weights')
        if not os.path.exists('gfpgan/weights/GFPGANv1.4.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./gfpgan/weights')
        if not os.path.exists('gfpgan/weights/RestoreFormer.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P ./gfpgan/weights'
            )

        # background enhancer with RealESRGAN
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_path = 'gfpgan/weights/realesr-general-x4v3.pth'
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

        # Use GFPGAN for face enhancement
        self.face_enhancer = GFPGANer(
            model_path='gfpgan/weights/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)
        self.current_version = 'v1.4'

    def predict(
            self,
            img: Path = Input(description='Input'),
            version: str = Input(
                description='GFPGAN version. v1.3: better quality. v1.4: more details and better identity.',
                choices=['v1.2', 'v1.3', 'v1.4', 'RestoreFormer'],
                default='v1.4'),
            scale: float = Input(description='Rescaling factor', default=2),
    ) -> Path:
        weight = 0.5
        print(img, version, scale, weight)
        try:
            extension = os.path.splitext(os.path.basename(str(img)))[1]
            img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            elif len(img.shape) == 2:
                img_mode = None
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_mode = None

            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            if self.current_version != version:
                if version == 'v1.2':
                    self.face_enhancer = GFPGANer(
                        model_path='gfpgan/weights/GFPGANv1.2.pth',
                        upscale=2,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler)
                    self.current_version = 'v1.2'
                elif version == 'v1.3':
                    self.face_enhancer = GFPGANer(
                        model_path='gfpgan/weights/GFPGANv1.3.pth',
                        upscale=2,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler)
                    self.current_version = 'v1.3'
                elif version == 'v1.4':
                    self.face_enhancer = GFPGANer(
                        model_path='gfpgan/weights/GFPGANv1.4.pth',
                        upscale=2,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler)
                    self.current_version = 'v1.4'
                elif version == 'RestoreFormer':
                    self.face_enhancer = GFPGANer(
                        model_path='gfpgan/weights/RestoreFormer.pth',
                        upscale=2,
                        arch='RestoreFormer',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler)

            try:
                _, _, output = self.face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            except RuntimeError as error:
                print('Error', error)

            try:
                if scale != 2:
                    interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                    h, w = img.shape[0:2]
                    output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
            except Exception as error:
                print('wrong scale input.', error)

            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            # save_path = f'output/out.{extension}'
            # cv2.imwrite(save_path, output)
            out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
            cv2.imwrite(str(out_path), output)
        except Exception as error:
            print('global exception: ', error)
        finally:
            clean_folder('output')
        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
