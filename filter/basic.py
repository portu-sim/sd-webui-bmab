import os
from PIL import Image
from PIL import ImageEnhance

import sd_bmab
from sd_bmab import util
from sd_bmab.base import filter


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
		noise = self.get_noise_from_cache(0, processed.width, processed.height).convert('LA')
		noise = noise.convert('RGBA')
		blended = Image.blend(processed.convert('RGBA'), noise, alpha=0.1)
		return self.basic_process(blended.convert('RGB'))

	def process(self, context, image: Image, processed: Image, *args, **kwargs):
		print('-----FILTER BASIC-----')
		return self.basic_process(processed)

	def postprocess(self, context, *args, **kwargs):
		pass

	@staticmethod
	def get_noise_from_cache(seed, width, height):
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../cache'))
		cache_file = f'{path}/noise_{width}_{height}.png'
		if os.path.isfile(cache_file):
			return Image.open(cache_file)
		img = util.generate_noise(seed, width, height)
		img.save(cache_file)
		return img
