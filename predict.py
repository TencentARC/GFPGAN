import subprocess

subprocess.call(["sh", "./run_setup.sh"])

import warnings
import tempfile
import os
from pathlib import Path
import argparse
import glob

import shutil
from basicsr.utils import imwrite
import torch
import cv2
import cog
from realesrgan import RealESRGANer
from gfpgan import GFPGANer


class Predictor(cog.Predictor):
    def setup(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--upscale", type=int, default=2)
        parser.add_argument("--arch", type=str, default="clean")
        parser.add_argument("--channel", type=int, default=2)
        parser.add_argument(
            "--model_path",
            type=str,
            default="experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
        )
        parser.add_argument("--bg_upsampler", type=str, default="realesrgan")
        parser.add_argument("--bg_tile", type=int, default=400)
        parser.add_argument("--test_path", type=str, default="inputs/whole_imgs")
        parser.add_argument(
            "--suffix", type=str, default=None, help="Suffix of the restored faces"
        )
        parser.add_argument("--only_center_face", action="store_true")
        parser.add_argument("--aligned", action="store_true")
        parser.add_argument("--paste_back", action="store_false")
        parser.add_argument("--save_root", type=str, default="results")

        self.args = parser.parse_args(
            ["--upscale", "2", "--test_path", "cog_temp", "--save_root", "results"]
        )
        os.makedirs(self.args.test_path, exist_ok=True)
        # background upsampler
        if self.args.bg_upsampler == "realesrgan":
            if not torch.cuda.is_available():  # CPU

                warnings.warn(
                    "The unoptimized RealESRGAN is very slow on CPU. We do not use it. "
                    "If you really want to use it, please modify the corresponding codes."
                )
                bg_upsampler = None
            else:
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path="https://github.com/xinntao/Real-ESRGAN/releases"
                    "/download/v0.2.1/RealESRGAN_x2plus.pth",
                    tile=self.args.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True,
                )  # need to set False in CPU mode
        else:
            bg_upsampler = None

        # set up GFPGAN restorer
        self.restorer = GFPGANer(
            model_path=self.args.model_path,
            upscale=self.args.upscale,
            arch=self.args.arch,
            channel_multiplier=self.args.channel,
            bg_upsampler=bg_upsampler,
        )

    @cog.input("image", type=Path, help="input image")
    def predict(self, image):
        try:
            input_dir = self.args.test_path

            input_path = os.path.join(input_dir, os.path.basename(image))
            shutil.copy(str(image), input_path)

            os.makedirs(self.args.save_root, exist_ok=True)

            img_list = sorted(glob.glob(os.path.join(input_dir, "*")))

            out_path = Path(tempfile.mkdtemp()) / "output.png"

            for img_path in img_list:
                # read image
                img_name = os.path.basename(img_path)
                print(f"Processing {img_name} ...")
                basename, ext = os.path.splitext(img_name)
                input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                    input_img,
                    has_aligned=self.args.aligned,
                    only_center_face=self.args.only_center_face,
                    paste_back=self.args.paste_back,
                )

                imwrite(restored_img, str(out_path))
                clean_folder(self.args.test_path)

                # save faces
                for idx, (cropped_face, restored_face) in enumerate(
                    zip(cropped_faces, restored_faces)
                ):
                    # save cropped face
                    save_crop_path = os.path.join(
                        self.args.save_root, "cropped_faces", f"{basename}_{idx:02d}.png"
                    )
                    imwrite(cropped_face, save_crop_path)
                    # save restored face
                    if self.args.suffix is not None:
                        save_face_name = f"{basename}_{idx:02d}_{self.args.suffix}.png"
                    else:
                        save_face_name = f"{basename}_{idx:02d}.png"
                    save_restore_path = os.path.join(
                        self.args.save_root, "restored_faces", save_face_name
                    )
                    imwrite(restored_face, save_restore_path)
                    imwrite(restored_img, str(out_path))
        finally:
            clean_folder(self.args.test_path)

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
            print("Failed to delete %s. Reason: %s" % (file_path, e))
