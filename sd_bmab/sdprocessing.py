from typing import Any

from modules.processing import StableDiffusionProcessingImg2Img


class StableDiffusionProcessingImg2ImgOv(StableDiffusionProcessingImg2Img):

	def __init__(self, init_images: list = None, resize_mode: int = 0, denoising_strength: float = 0.75,
				 image_cfg_scale: float = None, mask: Any = None, mask_blur: int = None, mask_blur_x: int = 4,
				 mask_blur_y: int = 4, inpainting_fill: int = 0, inpaint_full_res: bool = True,
				 inpaint_full_res_padding: int = 0, inpainting_mask_invert: int = 0,
				 initial_noise_multiplier: float = None, **kwargs):
		super().__init__(init_images, resize_mode, denoising_strength, image_cfg_scale, mask, mask_blur, mask_blur_x,
						 mask_blur_y, inpainting_fill, inpaint_full_res, inpaint_full_res_padding,
						 inpainting_mask_invert, initial_noise_multiplier, **kwargs)

		self.block_tqdm = False
