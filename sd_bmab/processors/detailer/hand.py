import math

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from modules import shared
from modules import devices

from sd_bmab import util
from sd_bmab.base import process_img2img, Context, ProcessorBase, VAEMethodOverride

from sd_bmab.util import debug_print
from sd_bmab.base.dino import dino_init, dino_predict


class Obj(object):
	name = None

	def __init__(self, xyxy) -> None:
		super().__init__()
		self.parent = None
		self.xyxy = xyxy
		self.objects = []
		self.inbox = xyxy

	def is_in(self, obj) -> bool:
		x1, y1, x2, y2 = self.inbox
		mx1, my1, mx2, my2 = obj.xyxy
		return mx1 <= x1 <= mx2 and mx1 <= x2 <= mx2 and my1 <= y1 <= my2 and my1 <= y2 <= my2

	def append(self, obj):
		obj.parent = self
		for ch in self.objects:
			if obj.is_in(ch):
				obj.parent = ch
				break
		self.objects.append(obj)

	def is_valid(self):
		return True

	def size(self):
		x1, y1, x2, y2 = self.xyxy
		return (x2 - x1) * (y2 - y1)

	def put(self, mask):
		for xg in self.objects:
			if not xg.is_valid():
				continue
			if xg.name == 'hand':
				dr = ImageDraw.Draw(mask, 'L')
				dr.rectangle(xg.xyxy, fill=255)

	def get_box(self):
		if not self.objects:
			return self.xyxy

		x1, y1, x2, y2 = self.xyxy
		ret = [x2, y2, x1, y1]
		for xg in self.objects:
			if not xg.is_valid():
				continue
			x = xg.xyxy
			ret[0] = x[0] if x[0] < ret[0] else ret[0]
			ret[1] = x[1] if x[1] < ret[1] else ret[1]
			ret[2] = x[2] if x[2] > ret[2] else ret[2]
			ret[3] = x[3] if x[3] > ret[3] else ret[3]

		return x1, y1, x2, ret[3]

	def log(self):
		debug_print(self.name, self.xyxy)
		for x in self.objects:
			x.log()


class Person(Obj):
	name = 'person'

	def __init__(self, xyxy, dilation) -> None:
		super().__init__(xyxy)
		self.inbox = util.fix_box_by_scale(xyxy, dilation)

	def is_valid(self):
		face = False
		hand = False
		for xg in self.objects:
			if xg.name == 'face':
				face = True
			if xg.name == 'hand':
				hand = True
		return face and hand

	def cleanup(self):
		debug_print([xg.name for xg in self.objects])
		nw = []
		for xg in self.objects:
			if xg.name == 'person':
				if len(self.objects) == 1 and xg.is_valid():
					self.xyxy = xg.xyxy
					self.objects = xg.objects
					return
				else:
					self.objects.extend(xg.objects)
			else:
				nw.append(xg)
		self.objects = nw


class Head(Obj):
	name = 'head'


class Face(Obj):
	name = 'face'


class Hand(Obj):
	name = 'hand'


def get_subframe(pilimg, dilation, box_threshold=0.30, text_threshold=0.20):
	text_prompt = "person . head . face . hand ."
	debug_print('threshold', box_threshold)
	boxes, logits, phrases = dino_predict(pilimg, text_prompt, box_threshold, text_threshold)

	people = []

	def find_person(o):
		for person in people:
			if o.is_in(person):
				return person
		return None

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		if phrase == 'person':
			p = Person(tuple(int(x) for x in box), dilation)
			parent = find_person(p)
			if parent:
				parent.append(p)
			else:
				people.append(p)
	people = sorted(people, key=lambda c: c.size(), reverse=True)

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		debug_print(float(logit), phrase)
		bb = tuple(int(x) for x in box)

		if phrase == 'head':
			o = Head(bb)
			parent = find_person(o)
			if parent:
				parent.append(o)
		elif phrase == 'face' or phrase == 'head face':
			o = Face(bb)
			parent = find_person(o)
			if parent:
				parent.append(o)
		elif phrase == 'hand':
			o = Hand(bb)
			parent = find_person(o)
			if parent:
				parent.append(o)

	for person in people:
		person.cleanup()

	boxes = []
	masks = []
	for person in people:
		if person.is_valid():
			mask = Image.new('L', pilimg.size, color=0)
			person.log()
			person.put(mask)
			boxes.append(person.get_box())
			masks.append(mask)
	return boxes, masks



class HandDetailer(ProcessorBase):
	def __init__(self) -> None:
		super().__init__()
		self.hand_detailing = None
		self.detailing_opt = None
		self.dilation = 0.1
		self.box_threshold = 0.3
		self.detailing_method = ''
		self.best_quality = False
		self.block_overscaled_image = True
		self.auto_upscale = True
		self.scale = 2
	def preprocess(self, context: Context, image: Image):
		if context.args['hand_detailing_enabled']:
			self.hand_detailing = dict(context.args.get('module_config', {}).get('hand_detailing', {}))
			self.detailing_opt = context.args.get('module_config', {}).get('hand_detailing_opt', {})
			self.dilation = self.hand_detailing.get('dilation', self.dilation)
			self.box_threshold = self.hand_detailing.get('box_threshold', self.box_threshold)
			self.detailing_method = self.detailing_opt.get('detailing_method', self.detailing_method)
			self.best_quality = self.detailing_opt.get('best_quality', self.best_quality)
			self.block_overscaled_image = self.detailing_opt.get('block_overscaled_image', self.block_overscaled_image)
			self.auto_upscale = self.detailing_opt.get('auto_upscale', self.auto_upscale)
			self.scale = self.detailing_opt.get('scale', self.scale)

		return context.args['hand_detailing_enabled']

	def process(self, context: Context, image: Image):

		dino_init()

		context.add_generation_param('BMAB_hand_option', util.dict_to_str(self.detailing_opt))
		context.add_generation_param('BMAB_hand_parameter', util.dict_to_str(self.hand_detailing))

		if self.detailing_method == 'subframe':
			return self.process_hand_detailing_subframe(context, image)
		elif self.detailing_method == 'at once':
			mask = Image.new('L', image.size, 0)
			dr = ImageDraw.Draw(mask, 'L')
			boxes, logits, phrases = dino_predict(image, 'person . hand')
			for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
				if phrase == 'hand':
					b = util.fix_box_size(box)
					dr.rectangle(b, fill=255)
			options = dict(mask=mask)
			options.update(self.hand_detailing)
			shared.state.job_count += 1
			with VAEMethodOverride():
				image = process_img2img(context.sdprocessing, image, options=options)
		elif self.detailing_method == 'each hand' or self.detailing_method == 'inpaint each hand':
			boxes, logits, phrases = dino_predict(image, 'person . hand')
			for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
				debug_print(float(logit), phrase)
				if phrase != 'hand':
					continue

				x1, y1, x2, y2 = tuple(int(x) for x in box)

				width = x2 - x1
				height = y2 - y1

				mbox = (int(x1 - width), int(y1 - height), int(x2 + width), int(y2 + height))
				mbox = util.fix_box_size(mbox)
				debug_print(mbox)

				hbox = (width, height, width * 2, height * 2)
				cropped_hand = image.crop(box=mbox)
				cropped_hand_mask = Image.new('L', cropped_hand.size, 0)
				dr = ImageDraw.Draw(cropped_hand_mask, 'L')
				dr.rectangle(hbox, fill=255)

				options = dict(seed=-1)
				scale = self.detailing_opt.get('scale', -1)
				if scale < 1:
					normalize = self.detailing_opt.get('normalize', 768)
					if width > height:
						scale = normalize / cropped_hand.width
					else:
						scale = normalize / cropped_hand.height
				if self.detailing_method == 'inpaint each hand':
					options['mask'] = cropped_hand_mask

				options.update(self.hand_detailing)
				w, h = util.fix_size_by_scale(cropped_hand.width, cropped_hand.height, scale)
				options['width'] = w
				options['height'] = h
				debug_print(f'scale {scale} width {w} height {h}')
				shared.state.job_count += 1
				with VAEMethodOverride(hiresfix=self.best_quality):
					img2img_result = process_img2img(context.sdprocessing, cropped_hand, options=options)
				img2img_result = img2img_result.resize(cropped_hand.size, resample=util.LANCZOS)

				debug_print('resize to', img2img_result.size, cropped_hand_mask.size)
				blur = ImageFilter.GaussianBlur(3)
				cropped_hand_mask = cropped_hand_mask.filter(blur)
				image.paste(img2img_result, (mbox[0], mbox[1]), mask=cropped_hand_mask)
		else:
			debug_print('no such method')
			return image

		return image

	def process_hand_detailing_subframe(self, context, image):

		boxes, masks = get_subframe(image, self.dilation, box_threshold=self.box_threshold)
		if not boxes:
			return image

		if not hasattr(context, 'hand_mask_image'):
			c1 = image.copy()
			for box, mask in zip(boxes, masks):
				box = util.fix_box_by_scale(box, self.dilation)
				draw = ImageDraw.Draw(c1, 'RGBA')
				draw.rectangle(box, outline=(0, 255, 0, 255), fill=(0, 255, 0, 50), width=3)
				c2 = image.copy()
				draw = ImageDraw.Draw(c2, 'RGBA')
				draw.rectangle(box, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50), width=3)
				c1.paste(c2, mask=mask)
			context.script.extra_image.append(c1)
			context.hand_mask_image = c1

		for box, mask in zip(boxes, masks):
			box = util.fix_box_by_scale(box, self.dilation)
			box = util.fix_box_size(box)
			box = util.fix_box_limit(box, image.size)
			x1, y1, x2, y2 = box

			cropped = image.crop(box=box)
			cropped_mask = mask.crop(box=box)

			options = dict(mask=cropped_mask, seed=-1)
			options.update(self.hand_detailing)
			w, h = util.fix_size_by_scale(cropped.width, cropped.height, self.scale)
			options['width'] = w
			options['height'] = h
			debug_print(f'Scale x{self.scale} ({cropped.width},{cropped.height}) -> ({w},{h})')

			if self.block_overscaled_image:
				area_org = context.get_max_area()
				area_scaled = w * h
				if area_scaled > area_org:
					debug_print(f'It is too large to process.')
					if not self.auto_upscale:
						context.add_generation_param(
							'BMAB_hand_SKIP', f'Image too large to process {cropped.width}x{cropped.height} {w}x{h}')
						return image
					new_scale = math.sqrt(area_org / (cropped.width * cropped.height))
					w, h = util.fix_size_by_scale(cropped.width, cropped.height, new_scale)
					options['width'] = w
					options['height'] = h
					debug_print(f'Auto Scale x{new_scale} ({cropped.width},{cropped.height}) -> ({w},{h})')
					if new_scale < 1.05:
						debug_print(f'Scale {new_scale} has no effect. skip!!!!!')
						context.add_generation_param('BMAB_hand_SKIP', f'{new_scale} < 1.2')
						return image
			shared.state.job_count += 1
			with VAEMethodOverride():
				img2img_result = process_img2img(context.sdprocessing, cropped, options=options)
			img2img_result = img2img_result.resize((cropped.width, cropped.height), resample=util.LANCZOS)
			blur = ImageFilter.GaussianBlur(3)
			cropped_mask = cropped_mask.filter(blur)
			image.paste(img2img_result, (x1, y1), mask=cropped_mask)
			devices.torch_gc()

		return image

	def postprocess(self, context: Context, image: Image):
		pass
