import os
import torch
from PIL import Image
from ultralytics import YOLO

import modules
from modules.paths import models_path

from sd_bmab.base.context import Context
from sd_bmab.base.detectorbase import DetectorBase
from sd_bmab.util import debug_print

from sd_bmab.base.dino import dino_init, dino_predict


class FaceDetector(DetectorBase):

	def description(self):
		return f'Face detecting using {self.target()}'


class GroundingDinoFaceDetector(FaceDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.box_threshold = kwargs.get('box_threshold', 0.35)
		self.text_threshold = kwargs.get('text_threshold', 0.25)

	def target(self):
		return 'GroundingDINO(face)'

	def predict(self, context: Context, image: Image):
		dino_init()
		boxes, logits, phrases = dino_predict(image, 'people . face .', box_threahold=self.box_threshold)
		debug_print(phrases)

		retboxes = []
		retlogits = []
		for box, logit, phrase in zip(boxes, logits, phrases):
			if phrase != 'face':
				continue
			retboxes.append(box)
			retlogits.append(logit)

		return retboxes, retlogits


class UltralyticsFaceDetector(FaceDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.confidence = kwargs.get('box_threshold', 0.35)
		self.model = None

	def target(self):
		return f'Ultralytics({self.model})'

	def predict(self, context: Context, image: Image):
		bmab_model_path = os.path.join(models_path, "bmab")
		yolo = f'{bmab_model_path}/{self.model}'
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


class UltralyticsFaceDetector8n(UltralyticsFaceDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'face_yolov8n.pt'


class UltralyticsFaceDetector8m(UltralyticsFaceDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'face_yolov8m.pt'


class UltralyticsFaceDetector8nv2(UltralyticsFaceDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'face_yolov8n_v2.pt'


class UltralyticsFaceDetector8s(UltralyticsFaceDetector):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model = 'face_yolov8s.pt'
