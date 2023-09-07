import torch
from PIL import Image
from ultralytics import YOLO

import modules

from sd_bmab import util
from sd_bmab.base.context import Context
from sd_bmab.base.detectorbase import DetectorBase
from sd_bmab.util import debug_print
from sd_bmab.base.dino import dino_init, dino_predict


class HandDetector(DetectorBase):

	def description(self):
		return f'Hand detecting using {self.target()}'


class GroundingDinoHandDetector(HandDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.box_threshold = kwargs.get('box_threshold', 0.35)
		self.text_threshold = kwargs.get('text_threshold', 0.25)

	def target(self):
		return 'GroundingDINO(hand)'

	def predict(self, context: Context, image: Image):
		dino_init()
		boxes, logits, phrases = dino_predict(image, 'people . hand .', box_threahold=self.box_threshold)
		debug_print(phrases)

		retboxes = []
		retlogits = []
		for box, logit, phrase in zip(boxes, logits, phrases):
			if phrase != 'hand':
				continue
			retboxes.append(box)
			retlogits.append(logit)

		return retboxes, retlogits


class UltralyticsHandDetector(HandDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.confidence = kwargs.get('box_threshold', 0.35)
		self.model = None

	def target(self):
		return f'Ultralytics({self.model})'

	def predict(self, context: Context, image: Image):
		yolo = util.lazy_loader(self.model)
		boxes = []
		confs = []
		load = torch.load
		torch.load = modules.safe.unsafe_torch_load
		try:
			model = YOLO(yolo)
			pred = model(image, conf=self.confidence, device='')
			boxes = pred[0].boxes.xyxy.cpu().numpy()
			boxes = boxes.tolist()
			confs.append(float(pred[0].boxes.conf))
		except:
			pass
		torch.load = load
		return boxes, confs


class UltralyticsHandDetector8n(UltralyticsHandDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'hand_yolov8n.pt'


class UltralyticsHandDetector8s(UltralyticsHandDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'hand_yolov8s.pt'
