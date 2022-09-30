import gradio as gr
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from transformers import logging

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

        sd = Pipeline.load_model_from_config(f"{runtime_cfg.ckpt.t2i}")
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

    def run(self):
        with gr.Blocks() as page:
            with gr.Tab("txt to img"):
                self.txt2img_interface()
            with gr.Tab("img to img"):
                self.img2img_interface()
            with gr.Tab("inpaint"):
                self.inpaint_interface()
        page.launch()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
