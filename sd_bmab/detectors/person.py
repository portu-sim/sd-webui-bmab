import torch
from PIL import Image
from ultralytics import YOLO

import modules

from sd_bmab import util
from sd_bmab.base.context import Context
from sd_bmab.base.detectorbase import DetectorBase


class PersonDetector(DetectorBase):

	def description(self):
		return f'Person detecting using {self.target()}'


class UltralyticsPersonDetector(PersonDetector):
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
			confs = pred[0].boxes.conf.tolist()
		except:
			pass
		torch.load = load
		return boxes, confs


class UltralyticsPersonDetector8n(UltralyticsPersonDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'person_yolov8n-seg.pt'


class UltralyticsPersonDetector8m(UltralyticsPersonDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'person_yolov8m-seg.pt'


class UltralyticsPersonDetector8s(UltralyticsPersonDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'person_yolov8s-seg.pt'
