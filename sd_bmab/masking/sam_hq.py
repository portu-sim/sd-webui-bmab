from sd_bmab.base import MaskBase
import cv2
import numpy as np

import torch

from PIL import Image
from modules.safe import unsafe_torch_load, load
from modules.devices import device, torch_gc

from segment_anything_hq import SamPredictor
from segment_anything_hq import sam_model_registry

from sd_bmab import util


sam_hq_model = None


class SamHqPredict(MaskBase):

	def __init__(self) -> None:
		super().__init__()

	@property
	def name(self):
		return f'sam_hq_{self.type}'

	@property
	def type(self):
		pass

	@property
	def file(self):
		pass

	@classmethod
	def init(cls, model_type, filename, *args, **kwargs):
		checkpoint_file = util.lazy_loader(filename)
		global sam_hq_model
		if sam_hq_model is None:
			torch.load = unsafe_torch_load
			sam_hq_model = sam_model_registry[model_type](checkpoint=checkpoint_file)
			sam_hq_model.to(device=device)
			torch.load = load
		return sam_hq_model

	def load(self):
		return SamHqPredict.init(self.type, self.file)

	def predict(self, image, boxes):
		sam = self.load()
		mask_predictor = SamPredictor(sam)

		numpy_image = np.array(image)
		opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		mask_predictor.set_image(opencv_image)

		result = Image.new('L', image.size, 0)
		for box in boxes:
			x1, y1, x2, y2 = box

			box = np.array([int(x1), int(y1), int(x2), int(y2)])
			masks, scores, logits = mask_predictor.predict(
				box=box,
				multimask_output=False
			)

			mask = Image.fromarray(masks[0])
			result.paste(mask, mask=mask)
		return result

	def predict_multiple(self, image, points, labels, boxes=None):
		sam = self.load()
		mask_predictor = SamPredictor(sam)

		numpy_image = np.array(image)
		opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		mask_predictor.set_image(opencv_image)

		result = Image.new('L', image.size, 0)
		kwargs = dict(multimask_output=True)
		if len(points) > 0:
			kwargs['point_coords'] = points
			kwargs['point_labels'] = labels
		if boxes is not None:
			for box in boxes:
				x1, y1, x2, y2 = box
				box = np.array([int(x1), int(y1), int(x2), int(y2)])
				masks, scores, logits = mask_predictor.predict(
					box=box,
					**kwargs
				)
				for mask in masks:
					mask = Image.fromarray(mask)
					result.paste(mask, mask=mask)
		else:
			masks, scores, logits = mask_predictor.predict(
				**kwargs
			)
			for mask in masks:
				mask = Image.fromarray(mask)
				result.paste(mask, mask=mask)
		return result

	@classmethod
	def release(cls):
		global sam_hq_model
		if sam_hq_model is not None:
			sam_hq_model = None
			torch_gc()


class SamHqPredictVitB(SamHqPredict):

	@property
	def type(self):
		return 'vit_b'

	@property
	def file(self):
		return 'sam_hq_vit_b.pth'
