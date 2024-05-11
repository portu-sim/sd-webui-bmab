import os
from PIL import Image
from PIL import ImageEnhance

import sd_bmab
from sd_bmab import util
from sd_bmab.base import filter
from sd_bmab.base import cache


class Filter(filter.BaseFilter):

	def preprocess(self, context, image, *args, **kwargs):
		pass

	def basic_process(self, image: Image):
		enhancer = ImageEnhance.Brightness(image)
		image = enhancer.enhance(0.8)
		enhancer = ImageEnhance.Contrast(image)
		image = enhancer.enhance(1.2)
		return image

	def basic_process_with_noise(self, processed: Image):
		noise = cache.get_noise_from_cache(0, processed.width, processed.height).convert('LA')
		noise = noise.convert('RGBA')
		blended = Image.blend(processed.convert('RGBA'), noise, alpha=0.1)
		return self.basic_process(blended.convert('RGB'))

	def process(self, context, image: Image, processed: Image, *args, **kwargs):
		print('-----FILTER BASIC-----')
		return self.basic_process(processed)

	def postprocess(self, context, *args, **kwargs):
		pass
