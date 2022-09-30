import gradio as gr
import numpy as np
import torch
from torchvision.utils import make_grid
from einops import rearrange
import os, re
from PIL import Image
import torch
import pandas as pd
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from optimizedSD.optimUtils import split_weighted_subprompts, logger
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
        ckpt = r"D:\diffusion\checkpoints\sd-v1-4.ckpt"
        sd = Pipeline.load_model_from_config(f"{ckpt}")
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

    def run(self):
        gr.TabbedInterface(
            [txt2img.demo, img2img.demo, inpaint.demo],
            ['txt to img', 'img to img', 'inpaint']
        ).launch()

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()