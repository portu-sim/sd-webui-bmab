from modules.processing import StableDiffusionProcessingImg2Img


class StableDiffusionProcessingImg2ImgOv(StableDiffusionProcessingImg2Img):

    def __post_init__(self):
        super().__post_init__()
        self.extra_noise = 0

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        return super().sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)

