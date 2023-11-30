
class MaskBase(object):
	def __init__(self) -> None:
		super().__init__()

	@property
	def name(self):
		pass

	@classmethod
	def init(cls, *args, **kwargs):
		pass

	def predict(self, image, box):
		pass

	def predict_multiple(self, image, points, labels, boxes=None):
		pass

	@classmethod
	def release(cls):
		pass
