import os
from PIL import Image

import sd_bmab
from sd_bmab import util


def get_noise_from_cache(seed, width, height):
	path = os.path.dirname(sd_bmab.__file__)
	path = os.path.normpath(os.path.join(path, '../resources/cache'))
	cache_file = f'{path}/noise_{width}_{height}.png'
	if os.path.isfile(cache_file):
		return Image.open(cache_file)
	img = util.generate_noise(seed, width, height)
	img.save(cache_file)
	return img


def get_image_from_cache(filename):
	path = os.path.dirname(sd_bmab.__file__)
	path = os.path.normpath(os.path.join(path, '../resources/cache'))
	full_path = os.path.join(path, filename)
	if os.path.exists(full_path):
		return Image.open(full_path)
	return None


def put_image_to_cache(filename, image):
	path = os.path.dirname(sd_bmab.__file__)
	path = os.path.normpath(os.path.join(path, '../resources/cache'))
	full_path = os.path.join(path, filename)
	image.save(full_path)
