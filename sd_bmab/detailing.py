from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw

from modules import devices
from modules.processing import StableDiffusionProcessingImg2Img
from sd_bmab import dinosam, util, process


def get_mask(img, prompt):
	boxes, logits, phrases = dinosam.dino_predict(img, prompt)
	sam_mask = dinosam.sam_predict_box(img, boxes[0])
	return sam_mask


def process_face_detailing(image, s, p, a):
	if a['face_detailing_enabled'] or a.get('module_config', {}).get('multiple_face'):
		return process_face_detailing_inner(image, s, p, a)
	return image


def process_face_detailing_old(images, s, p, a):
	if a['face_detailing_enabled'] or a.get('module_config', {}).get('multiple_face'):
		for idx in range(0, len(images)):
			pidx = p.iteration * p.batch_size + idx
			a['current_prompt'] = p.all_prompts[pidx]
			image = util.tensor_to_image(images[idx])
			image = process_face_detailing_inner(image, s, p, a)
			images[idx] = util.image_to_tensor(image)


def process_face_detailing_inner(image, s, p, a):
	multiple_face = a.get('module_config', {}).get('multiple_face', [])
	if multiple_face:
		return process_multiple_face(image, s, p, a)

	dilation = a.get('module_config', {}).get('face_detailing_opt', {}).get('mask dilation', 4)

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(image, 'face')
	# print(float(logits))
	print(phrases)

	org_size = image.size
	print('size', org_size)

	face_config = dict(a.get('module_config', {}).get('face_detailing', {}))

	prompt = face_config.get('prompt')
	current_prompt = a.get('current_prompt', '')
	if prompt is not None and prompt.find('#!org!#') >= 0:
		face_config['prompt'] = face_config['prompt'].replace('#!org!#', current_prompt)
		print('prompt for face', face_config['prompt'])

	for box, logit, phrase in zip(boxes, logits, phrases):
		print('render', phrase, float(logit))
		x1, y1, x2, y2 = box
		x1 = int(x1) - dilation
		y1 = int(y1) - dilation
		x2 = int(x2) + dilation
		y2 = int(y2) + dilation

		face_mask = Image.new('L', image.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle((x1, y1, x2, y2), fill=255)

		print('face lighting', a['face_lighting'])
		if a['face_lighting'] != 0:
			sam_mask = dinosam.sam_predict_box(image, box)
			enhancer = ImageEnhance.Brightness(image)
			bgimg = enhancer.enhance(1 + a['face_lighting'])
			image.paste(bgimg, mask=sam_mask)
			p.extra_generation_params['BMAB face lighting'] = a['face_lighting']

		options = dict(mask=face_mask, **face_config)
		image = process.process_img2img(p, image, options=options)
	devices.torch_gc()
	return image


def process_multiple_face(image, s, p, a):
	multiple_face = list(a.get('module_config', {}).get('multiple_face', []))
	order = a.get('module_config', {}).get('multiple_face_opt', {}).get('order', 'scale')
	dilation = a.get('module_config', {}).get('multiple_face_opt', {}).get('mask dilation', 4)
	limit = a.get('module_config', {}).get('multiple_face_opt', {}).get('limit', -1)

	print('processing multiple face')
	print(f'config : order={order}, dilation={dilation}, limit={limit}')

	if limit < 0:
		limit = len(multiple_face)
		if limit == 0:
			return image

	dinosam.dino_init()
	boxes, logits, phrases = dinosam.dino_predict(image, 'face')

	org_size = image.size
	print(f'size {org_size} boxes {len(boxes)} order {order}')

	# sort
	candidate = []
	for box, logit, phrase in zip(boxes, logits, phrases):
		x1, y1, x2, y2 = box
		if order == 'left':
			value = x1 + (x2 - x1) // 2
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif order == 'right':
			value = x1 + (x2 - x1) // 2
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		elif order == 'size':
			value = (x2 - x1) * (y2 - y1)
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		else:
			value = float(logit)
			print('detected', phrase, float(logit), value)
			candidate.append((value, box, logit, phrase))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

	for idx, (size, box, logit, phrase) in enumerate(candidate):
		if idx == limit:
			break
		print('render', phrase, float(logit), size)
		x1, y1, x2, y2 = box
		x1 = int(x1) - dilation
		y1 = int(y1) - dilation
		x2 = int(x2) + dilation
		y2 = int(y2) + dilation

		face_mask = Image.new('L', image.size, color=0)
		dr = ImageDraw.Draw(face_mask, 'L')
		dr.rectangle((x1, y1, x2, y2), fill=255)

		# face_mask = dinosam.sam_predict_box(img, box)

		options = dict(mask=face_mask)

		if idx < len(multiple_face):
			prompt = multiple_face[idx].get('prompt')
			current_prompt = a.get('current_prompt', '')
			if prompt is not None and prompt.find('#!org!#') >= 0:
				multiple_face[idx]['prompt'] = multiple_face[idx]['prompt'].replace('#!org!#', current_prompt)
				print('prompt for face', multiple_face[idx]['prompt'])
			options.update(multiple_face[idx])
		image = process.process_img2img(p, image, options=options)

	return image


def process_hand_detailing(image, s, p, a):
	if a['hand_detailing_enabled']:
		return process_hand_detailing_inner(image, s, p, a)
	return image


def process_hand_detailing_inner(image, s, p, args):
	hand_detailing = dict(args.get('module_config', {}).get('hand_detailing', {}))
	hand_detailing_opt = args.get('module_config', {}).get('hand_detailing_opt', {})
	detailing_method = hand_detailing_opt.get('detailing method', '')

	if not isinstance(p, StableDiffusionProcessingImg2Img):
		return

	image = p.init_images[0].copy()

	dinosam.dino_init()

	if detailing_method == 'subframe':
		return process_hand_detailing_subframe(image, s, p, args)
	elif detailing_method == 'at once':
		mask = Image.new('L', image.size, 0)
		dr = ImageDraw.Draw(mask, 'L')
		boxes, logits, phrases = dinosam.dino_predict(image, 'person . hand')
		for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
			if phrase == 'hand':
				dr.rectangle(tuple(int(x) for x in box), fill=255)
		options = dict(mask=mask)
		options.update(hand_detailing)
		image = process.process_img2img(p, image, options=options)
	elif detailing_method == 'each hand':
		boxes, logits, phrases = dinosam.dino_predict(image, 'person . hand')
		for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
			print(float(logit), phrase)
			if phrase == 'hand':
				x1, y1, x2, y2 = tuple(int(x) for x in box)

				width = x2 - x1
				height = y2 - y1

				mbox = (int(x1 - width), int(y1 - height), int(x2 + width), int(y2 + height))
				print(mbox)

				hbox = (width, height, width * 2, height * 2)
				hand = image.crop(box=mbox)
				hand_mask = Image.new('L', hand.size, 0)
				dr = ImageDraw.Draw(hand_mask, 'L')
				dr.rectangle(hbox, fill=255)

				options = {}
				scale = hand_detailing_opt.get('scale', -1)
				if scale < 1:
					normalize = hand_detailing_opt.get('normalize', 768)
					if width > height:
						scale = normalize / hand.width
					else:
						scale = normalize / hand.height
				mode = hand_detailing_opt.get('mode', -1)
				if mode == 'inpaint':
					options['mask'] = hand_mask

				options.update(hand_detailing)
				w = int(hand.width * scale)
				h = int(hand.height * scale)
				options['width'] = w
				options['height'] = h
				print(f'scale {scale} width {w} height {h}')
				hand_img = process.process_img2img(p, hand, options=options)
				hand = hand_img.resize((hand.width, hand.height), resample=Image.LANCZOS)

				print('resize to', (width, height))
				hand = hand.crop(box=hbox)
				image.paste(hand, box=(x1, y1))
	else:
		print('no such method')
		return image
	'''
	mask = Image.new('L', image.size, 0)
	dr = ImageDraw.Draw(mask, 'L')
	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		if phrase == 'hand':
			x1, y1, x2, y2 = box
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)
			dr.rectangle((x1, y1, x2, y2), fill=255)

	options = dict(denoising_strength=0.3, steps=20, mask=mask)
	image = process.process_img2img(p, image, options=options)
	'''

	return image


def process_hand_detailing_subframe(image, s, p, args):
	hand_detailing = dict(args.get('module_config', {}).get('hand_detailing', {}))
	hand_detailing_opt = args.get('module_config', {}).get('hand_detailing_opt', {})
	detailing_method = hand_detailing_opt.get('detailing method', '')

	boxes, mask = get_subframe(image)
	if not boxes:
		return image
	box = util.box_dilation(boxes[0], 0.1)
	x1, y1, x2, y2 = box

	if not hasattr(p, 'hand_mask_image'):
		c1 = image.copy()
		draw = ImageDraw.Draw(c1, 'RGBA')
		draw.rectangle(box, outline=(0, 255, 0, 255), fill=(0, 255, 0, 50), width=3)
		c2 = image.copy()
		draw = ImageDraw.Draw(c2, 'RGBA')
		draw.rectangle(box, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50), width=3)
		c1.paste(c2, mask=mask)
		s.extra_image.append(c1)
		p.hand_mask_image = c1

	'''
	subimg = image.crop(box=box)
	boxes2 = get_subframe(subimg)
	if boxes2:
		bx1, by1, bx2, by2 = boxes2[0]
		box = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
		c1 = image.copy()
		draw = ImageDraw.Draw(c1, 'RGBA')
		draw.rectangle(box, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50), width=3)
		s.extra_image.append(c1)
	'''
	cropped = image.crop(box=box)
	cropped_mask = mask.crop(box=box)

	scale = hand_detailing_opt.get('scale', 2)

	options = dict(mask=cropped_mask)
	hand_detailing = dict(args.get('module_config', {}).get('hand_detailing', {}))
	options.update(hand_detailing)
	options['width'] = cropped.width * scale
	options['height'] = cropped.height * scale

	img2img_result = process.process_img2img(p, cropped, options=options)
	img2img_result = img2img_result.resize((cropped.width, cropped.height), resample=Image.LANCZOS)
	image.paste(img2img_result, (x1, y1))

	return image


class Obj(object):
	name = None

	def __init__(self, xyxy) -> None:
		super().__init__()
		self.parent = None
		self.xyxy = xyxy
		self.objects = []

	def is_in(self, obj) -> bool:
		x1, y1, x2, y2 = self.xyxy
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
		print(self.name, self.xyxy)
		for x in self.objects:
			x.log()


class Person(Obj):
	name = 'person'

	def __init__(self, xyxy) -> None:
		super().__init__(xyxy)

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
		print([xg.name for xg in self.objects])
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


def get_subframe(pilimg, box_threshold=0.35, text_threshold=0.25):
	text_prompt = "person . head . face . hand ."

	boxes, logits, phrases = dinosam.dino_predict(pilimg, text_prompt, box_threshold, text_threshold)

	people = []

	def find_person(o):
		for person in people:
			if o.is_in(person):
				return person
		return None

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		if phrase == 'person':
			p = Person(tuple(int(x) for x in box))
			parent = find_person(p)
			if parent:
				parent.append(p)
			else:
				people.append(p)
	people = sorted(people, key=lambda c: c.size(), reverse=True)

	for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
		print(float(logit), phrase)
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

	mask = Image.new('L', pilimg.size, color=0)
	boxes = []
	for person in people:
		if person.is_valid():
			person.log()
			person.put(mask)
			boxes.append(person.get_box())

	return boxes, mask

