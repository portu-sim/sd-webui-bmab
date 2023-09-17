from PIL import Image

from modules import shared

from sd_bmab.base import dino
from sd_bmab.base.context import Context
from sd_bmab.base.processorbase import ProcessorBase
from sd_bmab import constants, util
from sd_bmab.util import debug_print


class IntermidiateResize(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.enabled = False
		self.resize_by_person_opt = None
		self.mode = constants.resize_mode_default
		self.value = 0

	def preprocess(self, context: Context, image: Image):
		self.enabled = context.args.get('resize_by_person_enabled', False)
		self.resize_by_person_opt = context.args.get('module_config', {}).get('resize_by_person_opt', {})
		self.mode = self.resize_by_person_opt.get('mode', self.mode)
		self.value = self.resize_by_person_opt.get('scale', self.value)

		if not self.enabled:
			return False
		if 0.79 > self.value >= 1.0:
			return False
		return self.enabled and self.mode == 'Intermediate'

	def process(self, context: Context, image: Image):
		context.add_generation_param('BMAB process_resize_by_person', self.value)
		debug_print('prepare dino')
		dino.dino_init()
		boxes, logits, phrases = dino.dino_predict(image, 'person')

		org_size = image.size
		debug_print('size', org_size)

		largest = (0, None)
		for box in boxes:
			x1, y1, x2, y2 = box
			size = (x2 - x1) * (y2 - y1)
			if size > largest[0]:
				largest = (size, box)

		if largest[0] == 0:
			return image

		x1, y1, x2, y2 = largest[1]
		ratio = (y2 - y1) / image.height
		debug_print('ratio', ratio)
		debug_print('org_size', org_size)

		if ratio > self.value:
			image_ratio = ratio / self.value
			if image_ratio < 1.0:
				return image
			debug_print('image resize ratio', image_ratio)
			image = util.resize_image(2, image, int(image.width * image_ratio), int(image.height * image_ratio))
			image = image.resize(org_size, resample=util.LANCZOS)
			context.add_generation_param('BMAB process_resize_by_person_ratio', '%.3s' % image_ratio)

		if shared.opts.bmab_optimize_vram != 'None':
			dino.release()
		return image

	def postprocess(self, context: Context, image: Image):
		pass
