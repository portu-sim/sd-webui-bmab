from dataclasses import dataclass

from modules.processing import StableDiffusionProcessingImg2Img


@dataclass(repr=False)
class StableDiffusionProcessingImg2ImgOv(StableDiffusionProcessingImg2Img):
    extra_noise: int = 0

    def __post_init__(self):
        super().__post_init__()

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        return super().sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)

