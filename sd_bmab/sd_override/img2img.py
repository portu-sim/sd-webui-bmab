from dataclasses import dataclass

from modules.processing import StableDiffusionProcessingImg2Img


@dataclass(repr=False)
class StableDiffusionProcessingImg2ImgOv(StableDiffusionProcessingImg2Img):
    extra_noise: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, all_prompts, all_seeds, all_subseeds):
        ret = super().init(all_prompts, all_seeds, all_subseeds)
        self.extra_generation_params['Hires prompt'] = ''
        self.extra_generation_params['Hires negative prompt'] = ''
        return ret
    
    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        return super().sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)

