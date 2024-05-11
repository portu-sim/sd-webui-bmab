import gc
import os
import sys
import torch
import importlib.util

import numpy as np
from PIL import Image

import modules
from modules import devices

import sd_bmab
from sd_bmab.base import cache
from sd_bmab.base import ProcessorBase
from sd_bmab.base.context import Context


class ICLight(ProcessorBase):

    def __init__(self) -> None:
        super().__init__()
        self.iclight_opt = {}
        self.enabled = False
        self.enable_before_upscale = False
        self.prompt = ''
        self.preference = 'None'
        self.use_background_image = False
        self.blending = 0.5

    def preprocess(self, context: Context, image: Image):
        self.iclight_opt = context.args.get('module_config', {}).get('iclight', {})
        self.enabled = self.iclight_opt.get('enabled', self.enabled)
        self.enable_before_upscale = self.iclight_opt.get('enable_before_upscale', self.enable_before_upscale)
        self.prompt = self.iclight_opt.get('prompt', self.prompt)
        self.preference = self.iclight_opt.get('preference', self.preference)
        self.use_background_image = self.iclight_opt.get('use_background_image', self.use_background_image)
        self.blending = self.iclight_opt.get('blending', self.blending)
        return self.enabled

    def process(self, context: Context, image: Image):
        load = torch.load
        torch.load = modules.safe.unsafe_torch_load
        try:
            if self.use_background_image:
                mod = ICLight.get_module('iclightbg')
                bg_image = self.get_background_image()
                img1 = image.convert('RGBA')
                img2 = bg_image.resize(img1.size, Image.LANCZOS).convert('RGBA')
                blended = Image.blend(img1, img2, alpha=self.blending)
                np_image = np.array(image.convert('RGB')).astype("uint8")
                input_bg = np.array(blended.convert('RGB')).astype("uint8")
                input_fg, matting = mod.run_rmbg(np_image)
                seed, subseed = context.get_seeds()
                result = mod.process_relight(input_fg, input_bg, self.prompt, image.width, image.height, 1, seed, 20,
                                                 'best quality', 'lowres, bad anatomy, bad hands, cropped, worst quality',
                                                 7, 1.5, 0.5, 'Use Background Image')
                context.add_extra_image(image)
                context.add_extra_image(bg_image)
                mod.clean_up()
                del mod
                return result

            else:
                mod = ICLight.get_module('iclightnm')
                np_image = np.array(image.convert('RGB')).astype("uint8")
                input_fg, matting = mod.run_rmbg(np_image)
                seed, subseed = context.get_seeds()
                result = mod.process_relight(input_fg, self.prompt, image.width, image.height, 1, seed, 25,
                                'best quality', 'lowres, bad anatomy, bad hands, cropped, worst quality',
                                2, 1.5, 0.5, 0.9, self.preference)
                context.add_extra_image(image)
                mod.clean_up()
                del mod
                return result
        finally:
            torch.load = load
            devices.torch_gc()

    def postprocess(self, context: Context, image: Image):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def get_background_image():
        img = cache.get_image_from_cache('iclight_background.png')
        if img is not None:
            return img
        return Image.new('RGB', (512, 768), (0, 0, 0))

    @staticmethod
    def put_backgound_image(img):
        cache.put_image_to_cache('iclight_background.png', img)

    @staticmethod
    def get_module(name):
        path = os.path.dirname(sd_bmab.__file__)
        path = os.path.normpath(os.path.join(path, f'external/iclight/{name}.py'))
        return ICLight.load_module(path, 'module')

    @staticmethod
    def load_module(file_name, module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


class ICLightBeforeUpsacle(ICLight):

    def preprocess(self, context: Context, image: Image):
        super().preprocess(context, image)
        return self.enabled and self.enable_before_upscale


class ICLightAfterUpsacle(ICLight):

    def preprocess(self, context: Context, image: Image):
        super().preprocess(context, image)
        return self.enabled and not self.enable_before_upscale
