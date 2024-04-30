import os
from PIL import Image
from PIL import ImageEnhance

import sd_bmab
from sd_bmab import util
from sd_bmab.base import filter
from sd_bmab.processors.basic import final


CONTRAST = 0.8
BRIGHTNESS = 0.9
SHARPNESS = 0.5
COLOR = 0.85
COLOR_TEMPERATURE = 5240
NOISE = 0.05

class Filter(filter.BaseFilter):

	def preprocess(self, context, image, *args, **kwargs):
		pass

	def basic_process(self, image: Image):
		enhancer = ImageEnhance.Contrast(image)
		image = enhancer.enhance(CONTRAST)
		enhancer = ImageEnhance.Brightness(image)
		image = enhancer.enhance(BRIGHTNESS)
		enhancer = ImageEnhance.Sharpness(image)
		image = enhancer.enhance(SHARPNESS)
		enhancer = ImageEnhance.Color(image)
		image = enhancer.enhance(COLOR)
		temp = final.calc_color_temperature(COLOR_TEMPERATURE)
		az = []
		for d in image.getdata():
			az.append((int(d[0] * temp[0]), int(d[1] * temp[1]), int(d[2] * temp[2])))
		image = Image.new('RGB', image.size)
		image.putdata(az)
		noise = self.get_noise_from_cache(0, image.size[0], image.size[1])
		image = Image.blend(image, noise, alpha=NOISE)
		return image

	def process(self, context, image: Image, processed: Image, *args, **kwargs):
		print('-----FILTER VINTAGE-----')
		return self.basic_process(processed)

	def postprocess(self, context, *args, **kwargs):
		pass

	@staticmethod
	def get_noise_from_cache(seed, width, height):
		path = os.path.dirname(sd_bmab.__file__)
		path = os.path.normpath(os.path.join(path, '../resources/cache'))
		cache_file = f'{path}/noise_{width}_{height}.png'
		if os.path.isfile(cache_file):
			return Image.open(cache_file)
		img = util.generate_noise(seed, width, height)
		img.save(cache_file)
		return img
