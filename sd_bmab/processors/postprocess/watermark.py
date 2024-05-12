import os
import sys
import glob

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sd_bmab.base import Context, ProcessorBase


class Watermark(ProcessorBase):
	alignment = {
		'top': lambda w, h, cx, cy: (w / 2 - cx / 2, 0),
		'top-right': lambda w, h, cx, cy: (w - cx, 0),
		'right': lambda w, h, cx, cy: (w - cx, h / 2 - cy / 2),
		'bottom-right': lambda w, h, cx, cy: (w - cx, h - cy),
		'bottom': lambda w, h, cx, cy: (w / 2 - cx / 2, h - cy),
		'bottom-left': lambda w, h, cx, cy: (0, h - cy),
		'left': lambda w, h, cx, cy: (0, h / 2 - cy / 2),
		'top-left': lambda w, h, cx, cy: (0, 0),
		'center': lambda w, h, cx, cy: (w / 2 - cx / 2, h / 2 - cy / 2),
	}

	def __init__(self) -> None:
		super().__init__()
		self.enabled = False
		self.font = None
		self.alignment = 'bottom-left'
		self.text_alignment = 'left'
		self.rotate = 0
		self.color = '#000000'
		self.background_color = '#00000000'
		self.font_size = 12
		self.transparency = 100
		self.background_transparency = 0
		self.margin = 5
		self.text = ''

	def preprocess(self, context: Context, image: Image):
		watermark_opt = context.args.get('module_config', {}).get('watermark', {})
		self.enabled = watermark_opt.get('enabled', self.enabled)
		self.font = watermark_opt.get('font', self.font)
		self.alignment = watermark_opt.get('alignment', self.alignment)
		self.text_alignment = watermark_opt.get('text_alignment', self.text_alignment)
		_rotate = watermark_opt.get('rotate', self.rotate)
		self.rotate = int(_rotate)
		self.color = watermark_opt.get('color', self.color)
		self.background_color = watermark_opt.get('background_color', self.background_color)
		self.font_size = watermark_opt.get('font_size', self.font_size)
		self.transparency = watermark_opt.get('transparency', self.transparency)
		self.background_transparency = watermark_opt.get('background_transparency', self.background_transparency)
		self.margin = watermark_opt.get('margin', self.margin)
		self.text = watermark_opt.get('text', self.text)

		return self.enabled

	def process(self, context: Context, image: Image):

		background_color = self.color_hex_to_rgb(self.background_color, int(255 * (self.background_transparency / 100)))

		if os.path.isfile(self.text):
			cropped = Image.open(self.text)
		else:
			font = self.get_font(self.font, self.font_size)
			color = self.color_hex_to_rgb(self.color, int(255 * (self.transparency / 100)))

			# 1st
			base = Image.new('RGBA', image.size, background_color)
			draw = ImageDraw.Draw(base)
			bbox = draw.textbbox((0, 0), self.text, font=font)
			draw.text((0, 0), self.text, font=font, fill=color, align=self.text_alignment)
			cropped = base.crop(bbox)

		# 2st margin
		base = Image.new('RGBA', (cropped.width + self.margin * 2, cropped.height + self.margin * 2), background_color)
		base.paste(cropped, (self.margin, self.margin))

		# 3rd rotate
		base = base.rotate(self.rotate, expand=True)

		# 4th
		image = image.convert('RGBA')
		image2 = image.copy()
		x, y = Watermark.alignment[self.alignment](image.width, image.height, base.width, base.height)
		image2.paste(base, (int(x), int(y)))
		image = Image.alpha_composite(image, image2)

		return image.convert('RGB')

	@staticmethod
	def color_hex_to_rgb(value, transparency):
		value = value.lstrip('#')
		lv = len(value)
		r, g, b = tuple(int(value[i:i + 2], 16) for i in range(0, lv, 2))
		return r, g, b, transparency

	@staticmethod
	def list_fonts():
		if sys.platform == 'win32':
			path = 'C:\\Windows\\Fonts\\*.ttf'
			files = glob.glob(path)
			return [os.path.basename(f) for f in files]
		if sys.platform == 'darwin':
			path = '/System/Library/Fonts/*'
			files = glob.glob(path)
			return [os.path.basename(f) for f in files]
		if sys.platform == 'linux':
			path = '/usr/share/fonts/*'
			files = glob.glob(path)
			fonts = [os.path.basename(f) for f in files]
			if 'SAGEMAKER_INTERNAL_IMAGE_URI' in os.environ:
				path = '/opt/conda/envs/sagemaker-distribution/fonts/*'
				files = glob.glob(path)
				fonts.extend([os.path.basename(f) for f in files])
			return fonts
		return ['']

	@staticmethod
	def get_font(font, size):
		if sys.platform == 'win32':
			path = f'C:\\Windows\\Fonts\\{font}'
			return ImageFont.truetype(path, size, encoding="unic")
		if sys.platform == 'darwin':
			path = f'/System/Library/Fonts/{font}'
			return ImageFont.truetype(path, size, encoding="unic")
		if sys.platform == 'linux':
			if 'SAGEMAKER_INTERNAL_IMAGE_URI' in os.environ:
				path = f'/opt/conda/envs/sagemaker-distribution/fonts/{font}'
			else:
				path = f'/usr/share/fonts/{font}'
			return ImageFont.truetype(path, size, encoding="unic")
