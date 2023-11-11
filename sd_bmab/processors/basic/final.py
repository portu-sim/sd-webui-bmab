import math

from PIL import Image
from PIL import ImageEnhance

from sd_bmab import util
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase


def calc_color_temperature(temp):
	white = (255.0, 254.11008387561782, 250.0419083427406)

	temperature = temp / 100

	if temperature <= 66:
		red = 255.0
	else:
		red = float(temperature - 60)
		red = 329.698727446 * math.pow(red, -0.1332047592)
		if red < 0:
			red = 0
		if red > 255:
			red = 255

	if temperature <= 66:
		green = temperature
		green = 99.4708025861 * math.log(green) - 161.1195681661
	else:
		green = float(temperature - 60)
		green = 288.1221695283 * math.pow(green, -0.0755148492)
	if green < 0:
		green = 0
	if green > 255:
		green = 255

	if temperature >= 66:
		blue = 255.0
	else:
		if temperature <= 19:
			blue = 0.0
		else:
			blue = float(temperature - 10)
			blue = 138.5177312231 * math.log(blue) - 305.0447927307
			if blue < 0:
				blue = 0
			if blue > 255:
				blue = 255

	return red / white[0], green / white[1], blue / white[2]


class FinalProcessorBasic(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.noise_alpha_final = 0
		self.contrast = 1
		self.brightness = 1
		self.sharpeness = 1
		self.color_saturation = 1
		self.color_temperature = 0

	def preprocess(self, context: Context, image: Image):
		self.contrast = context.args['contrast']
		self.brightness = context.args['brightness']
		self.sharpeness = context.args['sharpeness']
		self.color_saturation = context.args['color_saturation']
		self.color_temperature = context.args['color_temperature']
		self.noise_alpha_final = context.args['noise_alpha_final']
		return True

	def process(self, context: Context, image: Image):

		if self.noise_alpha_final != 0:
			context.add_generation_param('BMAB noise alpha final', self.noise_alpha_final)
			img_noise = util.generate_noise(image.size[0], image.size[1])
			image = Image.blend(image, img_noise, alpha=self.noise_alpha_final)

		if self.contrast != 1:
			context.add_generation_param('BMAB contrast', self.contrast)
			enhancer = ImageEnhance.Contrast(image)
			image = enhancer.enhance(self.contrast)

		if self.brightness != 1:
			context.add_generation_param('BMAB brightness', self.brightness)
			enhancer = ImageEnhance.Brightness(image)
			image = enhancer.enhance(self.brightness)

		if self.sharpeness != 1:
			context.add_generation_param('BMAB sharpeness', self.sharpeness)
			enhancer = ImageEnhance.Sharpness(image)
			image = enhancer.enhance(self.sharpeness)

		if self.color_saturation != 1:
			context.add_generation_param('BMAB color', self.color_saturation)
			enhancer = ImageEnhance.Color(image)
			image = enhancer.enhance(self.color_saturation)

		if self.color_temperature != 0:
			context.add_generation_param('BMAB color temperature', self.color_temperature)
			temp = calc_color_temperature(6500 + self.color_temperature)
			az = []
			for d in image.getdata():
				az.append((int(d[0] * temp[0]), int(d[1] * temp[1]), int(d[2] * temp[2])))
			image = Image.new('RGB', image.size)
			image.putdata(az)

		return image

	def postprocess(self, context: Context, image: Image):
		pass
