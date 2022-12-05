import sys
import os
import time
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from transformers import logging
from gfpgan import GFPGANer
from basicsr.utils import imwrite
from einops import rearrange

logging.set_verbosity_error()

import mimetypes

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

import txt2img_gradio as txt2img
import img2img_gradio as img2img
import inpaint_gradio as inpaint


class Pipeline():
    @staticmethod
    def load_model_from_config(ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        return sd

    def __init__(self):
        config = "optimizedSD/v1-inference.yaml"
        runtime_cfg = OmegaConf.load(f"runtime.yaml")

        sd = Pipeline.load_model_from_config(f"{runtime_cfg.ckpt}")
        self.gfpan_model = runtime_cfg.gfpan_model
        li, lo = [], []
        for key, v_ in sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd["model1." + key[6:]] = sd.pop(key)
        for key in lo:
            sd["model2." + key[6:]] = sd.pop(key)

        config = OmegaConf.load(f"{config}")

        model = instantiate_from_config(config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.eval()
        self.model = model

        modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.eval()
        self.modelCS = modelCS

        modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()
        self.modelFS = modelFS

        del sd

        txt2img.model = self.model
        txt2img.modelCS = self.modelCS
        txt2img.modelFS = self.modelFS

        img2img.model = self.model
        img2img.modelCS = self.modelCS
        img2img.modelFS = self.modelFS

        inpaint.model = self.model
        inpaint.modelCS = self.modelCS
        inpaint.modelFS = self.modelFS

    # region txt2img
    def run_txt2img(self,
                    prompt,
                    ddim_steps,
                    n_iter,
                    width,
                    height,
                    seed,
                    scale,
                    unet_bs,
                    turbo,
                    img_format,
                    outdir,
                    device,
                    sampler
                    ):
        return txt2img.generate(
            prompt,
            ddim_steps,
            n_iter,
            1,  # batch_size
            height,
            width,
            scale,
            0.0,  # ddim_eta,
            unet_bs,
            device,
            seed,
            outdir,
            img_format,
            turbo,
            False,  # full_precision,
            sampler,
        )

    def txt2img_interface(self):
        with gr.Blocks() as ret:
            with gr.Row():
                with gr.Column():
                    inputs = [
                        gr.Textbox(label="prompt 描述"),
                        gr.Slider(1, 1000, value=50, label="ddim_steps 迭代次数"),
                        gr.Slider(1, 100, step=1, label="n_iter 输出图片数"),
                        gr.Slider(64, 4096, value=512, step=64, label="width 图片宽度"),
                        gr.Slider(64, 4096, value=512, step=64, label="height 图片高度"),
                        gr.Textbox(value="", label="seed 起始种子"),
                        gr.Slider(0, 50, value=7.5, step=0.1, label="scale 和描述的相似度"),
                        gr.Slider(1, 2, value=1, step=1, label="unet_bs 消耗显存加速"),
                        gr.Checkbox(label="turbo 消耗显存加速"),
                        gr.Radio(["png", "jpg"], value='png', label="输出格式"),
                        gr.Text(value="outputs/txt2img-samples", label="outdir 输出目录"),
                        gr.Text(value="cuda", label="cuda/cpu"),
                        gr.Radio(["ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"], value="plms",
                                 label="sampler 算法")
                    ]

                with gr.Column():
                    outputs = [
                        gr.Image(label="output"),
                        gr.Text(label="output"),
                    ]

            with gr.Row():
                gr.Button("Go!").click(fn=self.run_txt2img, inputs=inputs, outputs=outputs)
                gr.Button("Stop!").click(fn=txt2img.stop, inputs=[], outputs=[])

        return ret

    # endregion

    # region img2img
    def run_img2img(self,
                    image,
                    prompt,
                    ddim_steps,
                    n_iter,
                    width,
                    height,
                    strength,
                    scale,
                    seed,
                    unet_bs,
                    turbo,
                    img_format,
                    outdir,
                    device,
                    ):
        return img2img.generate(
            image,
            prompt,
            strength,
            ddim_steps,
            n_iter,
            1,  # batch_size,
            height,
            width,
            scale,
            0.0,  # ddim_eta,
            unet_bs,
            device,
            seed,
            outdir,
            img_format,
            turbo,
            False,  # full_precision,
        )

    def img2img_interface(self):
        with gr.Blocks() as ret:
            with gr.Row():
                with gr.Column():
                    inputs = [
                        gr.Image(tool="editor", type="pil"),
                        gr.Textbox(label="prompt 描述"),
                        gr.Slider(1, 1000, value=50, label="ddim_steps 迭代次数"),
                        gr.Slider(1, 100, step=1, label="n_iter 输出图片数"),
                        gr.Slider(64, 4096, value=512, step=64, label="width 图片宽度"),
                        gr.Slider(64, 4096, value=512, step=64, label="height 图片高度"),
                        gr.Slider(0, 1, value=0.75, label="strength 和源图片的相似度"),
                        gr.Slider(0, 50, value=7.5, step=0.1, label="scale 和描述的相似度"),
                        gr.Textbox(value="", label="seed 起始种子"),
                        gr.Slider(1, 2, value=1, step=1, label="unet_bs 消耗显存加速"),
                        gr.Checkbox(label="turbo 消耗显存加速"),
                        gr.Radio(["png", "jpg"], value='png', label="输出格式"),
                        gr.Text(value="outputs/img2img-samples", label="outdir 输出目录"),
                        gr.Text(value="cuda", label="cuda/cpu"),
                    ]

                with gr.Column():
                    outputs = [
                        gr.Image(label="output"),
                        gr.Text(label="output"),
                    ]

            with gr.Row():
                gr.Button("Go!").click(fn=self.run_img2img, inputs=inputs, outputs=outputs)
                gr.Button("Stop!").click(fn=img2img.stop, inputs=[], outputs=[])

        return ret

    # endregion

    # region inpaint
    def run_inpaint(self,
                    image,
                    mask_image,
                    prompt,
                    ddim_steps,
                    n_iter,
                    width,
                    height,
                    strength,
                    scale,
                    seed,
                    unet_bs,
                    turbo,
                    outdir,
                    img_format,
                    device,
                    ):
        return inpaint.generate(
            image,
            mask_image,
            prompt,
            strength,
            ddim_steps,
            n_iter,
            1,
            height,
            width,
            scale,
            0.0,
            unet_bs,
            device,
            seed,
            outdir,
            img_format,
            turbo,
            False,
        )

    def inpaint_interface(self):
        with gr.Blocks() as ret:
            with gr.Row():
                with gr.Column():
                    inputs = [
                        gr.Image(tool="sketch", type="pil", label="源图像，可以直接涂抹"),
                        gr.Image(tool="editor", type="pil", label="蒙版"),
                        gr.Textbox(label="prompt 描述"),
                        gr.Slider(1, 1000, value=50, label="ddim_steps 迭代次数"),
                        gr.Slider(1, 100, step=1, label="n_iter 输出图片数"),
                        gr.Slider(64, 4096, value=512, step=64, label="width 图片宽度"),
                        gr.Slider(64, 4096, value=512, step=64, label="height 图片高度"),
                        gr.Slider(0, 1, value=0.75, label="strength 和源图片的相似度"),
                        gr.Slider(0, 50, value=7.5, step=0.1, label="scale 和描述的相似度"),
                        gr.Textbox(value="", label="seed 起始种子"),
                        gr.Slider(1, 2, value=1, step=1, label="unet_bs 消耗显存加速"),
                        gr.Checkbox(label="turbo 消耗显存加速"),
                        gr.Text(value="outputs/inpaint-samples", label="outdir 输出目录"),
                        gr.Radio(["png", "jpg"], value='png', label="输出格式"),
                        gr.Text(value="cuda", label="cuda/cpu"),
                    ]

                with gr.Column():
                    outputs = [
                        gr.Image(label="output"),
                        gr.Image(label="mask"),
                        gr.Text(label="output"),
                    ]

            with gr.Row():
                gr.Button("Go!").click(fn=self.run_inpaint, inputs=inputs, outputs=outputs)
                gr.Button("Stop!").click(fn=inpaint.stop, inputs=[], outputs=[])

        return ret    
    # endregion

    # region gfppan
    @staticmethod
    def get_arch(version):        
        if version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {version}.')
        return arch, channel_multiplier, model_name, url

    @staticmethod
    def get_bg_upsampler(bg_upsampler, bg_tile):
        # ------------------------ set up background upsampler ------------------------
        if bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')
                return None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                return RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            return None


    def run_gfppan(self, img_path, upscale, extension, output, save_faces, bg_tile, weight, version):
        restored_img = None

        # read image
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        bg_upsampler = "realesrgan"
        arch, channel_multiplier, model_name, url = Pipeline.get_arch(version)
        restorer = GFPGANer(
            model_path=self.gfpan_model,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=Pipeline.get_bg_upsampler(bg_upsampler, bg_tile)
        )

        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight
        )

        # save faces
        file_name = f'{int(time.time())}'
        if save_faces:
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):

                # save cropped face
                save_crop_path = os.path.join(output, 'cropped_faces', f'{file_name}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)

                # save restored face
                save_face_name = f'{file_name}_{idx:02d}.png'
                save_restore_path = os.path.join(output, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)

                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(output, 'cmp', f'{file_name}_{idx:02d}.png'))



        # save restored img
        save_restore_path = ''
        if restored_img is not None:
            if extension == 'auto':
                extension = ext[1:]

            save_restore_path = os.path.join(output, f'{file_name}.{extension}')
            imwrite(restored_img, save_restore_path)

        return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)), f'Saved as {save_restore_path}'

    def gfpgan_interface(self):
        with gr.Blocks() as ret:
            with gr.Row():
                with gr.Column():
                    inputs = [
                        gr.Image(tool="editor", type="filepath"),
                        gr.Slider(1, 10, step=1, label="upscale |放大倍数"),
                        gr.Radio(["auto", "png", "jpg"], value='auto', label="输出格式"),
                        gr.Text(value="outputs/upscale", label="outdir |输出目录"),
                        gr.Checkbox(label="单独存储识别到的人脸"),
                        gr.Slider(1, 1000, step=10, value=400, label="bg_tile |背景分块像素尺寸"),
                        gr.Slider(0, 1, value=0.5, label="weight |Adjustable weights."),
                        gr.Radio(["1", "1.2", "1.3", "1.4", "RestoreFormer"], value="1.3", label="version"),
                    ]

                with gr.Column():
                    outputs = [
                        gr.Image(label="output"),
                        gr.Text(label="output"),
                    ]
            with gr.Row():
                gr.Button("Go!").click(fn=self.run_gfppan, inputs=inputs, outputs=outputs)

    # endregion

    def run(self):
        with gr.Blocks() as page:
            with gr.Tab("txt to img"):
                self.txt2img_interface()
            with gr.Tab("img to img"):
                self.img2img_interface()
            with gr.Tab("inpaint"):
                self.inpaint_interface()
            with gr.Tab("upscale"):
                self.gfpgan_interface()
        page.launch()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
