import torch
import numpy as np
from PIL import Image

from modules import shared
from modules import devices
from modules import images
from modules.sd_samplers import sample_to_image
from scripts.dinosam import dino_init, dino_predict, sam_predict


def image_to_latent(p, img):
    image = np.array(img).astype(np.float32) / 255.0
    image = np.moveaxis(image, 2, 0)
    batch_images = np.expand_dims(image, axis=0).repeat(1, axis=0)
    image = torch.from_numpy(batch_images)
    image = 2. * image - 1.
    image = image.to(shared.device, dtype=devices.dtype_vae)
    return p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image))


def latent_to_image(x, index=0):
    img = sample_to_image(x, index, approximation=0)
    return img


def tensor_to_image(xx):
    x_sample = 255. * np.moveaxis(xx.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)
    return Image.fromarray(x_sample)


def image_to_tensor(xx):
    image = np.array(xx).astype(np.float32) / 255
    image = np.moveaxis(image, 2, 0)
    image = torch.from_numpy(image)
    return image


def resize_image(resize_mode, im, width, height, upscaler_name=None):
    if resize_mode == 2:
        vwidth = im.width
        vheight = height
        res = Image.new("RGB", (vwidth, vheight))
        dw = (vwidth - im.width) // 2
        dh = (vheight - im.height)
        res.paste(im, (dw, dh))
        if dh > 0:
            res.paste(im.resize((vwidth, dh), box=(0, 0, vwidth, 0)), box=(0, 0))

        im = res
        vwidth = width
        vheight = height
        res = Image.new("RGB", (vwidth, vheight))
        dw = (vwidth - im.width) // 2
        dh = (vheight - im.height)
        res.paste(im, (dw, dh))

        if dw > 0:
            res.paste(im.resize((dw, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(im.resize((dw, height), box=(im.width, 0, im.width, height)),
                      box=(im.width + dw, 0))

        return res

    return images.resize_image(resize_mode, im, width, height, upscaler_name)


def sam(prompt, input_image):
    boxes, logits, phrases = dino_predict(input_image, prompt, 0.35, 0.25)
    mask = sam_predict(input_image, boxes)
    return mask

