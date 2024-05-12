from PIL import Image

from sd_bmab.base import cache
from sd_bmab.base import ProcessorBase
from sd_bmab.base.context import Context
from sd_bmab.external import load_external_module


class ICLight(ProcessorBase):

    def __init__(self) -> None:
        super().__init__()
        self.iclight_opt = {}
        self.enabled = False
        self.enable_before_upscale = False
        self.style = 'normal'
        self.prompt = ''
        self.preference = 'None'
        self.use_background_image = False
        self.blending = 0.5

    def preprocess(self, context: Context, image: Image):
        self.iclight_opt = context.args.get('module_config', {}).get('iclight', {})
        self.enabled = self.iclight_opt.get('enabled', self.enabled)
        self.enable_before_upscale = self.iclight_opt.get('enable_before_upscale', self.enable_before_upscale)
        self.style = self.iclight_opt.get('style', self.style)
        self.prompt = self.iclight_opt.get('prompt', self.prompt)
        self.preference = self.iclight_opt.get('preference', self.preference)
        self.use_background_image = self.iclight_opt.get('use_background_image', self.use_background_image)
        self.blending = self.iclight_opt.get('blending', self.blending)
        return self.enabled

    def process(self, context: Context, image: Image):
        mod = load_external_module('iclight', 'bmabiclight')
        bg_image = self.get_background_image() if self.use_background_image else None
        return mod.bmab_relight(context, self.style, image, bg_image, self.prompt, self.blending, self.preference)

    def postprocess(self, context: Context, image: Image):
        pass

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
    def get_styles():
        return ['intensive', 'less intensive', 'normal', 'soft']


class ICLightBeforeUpsacle(ICLight):

    def preprocess(self, context: Context, image: Image):
        super().preprocess(context, image)
        return self.enabled and self.enable_before_upscale


class ICLightAfterUpsacle(ICLight):

    def preprocess(self, context: Context, image: Image):
        super().preprocess(context, image)
        return self.enabled and not self.enable_before_upscale
