import os
import platform

import launch
import glob
import torch

from modules.paths import models_path

from basicsr.utils.download_util import load_file_from_url
from packaging.version import parse


def install_models():
    bmab_model_path = os.path.join(models_path, "bmab")

    targets = {
        ('sam_vit_b_01ec64.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'),
        ('sam_vit_l_0b3195.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'),
        ('sam_vit_h_4b8939.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'),
        ('groundingdino_swint_ogc.pth', 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth'),
        ('GroundingDINO_SwinT_OGC.py', 'https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    }

    files = glob.glob(bmab_model_path)

    for target in targets:
        if target[0] not in files:
            load_file_from_url(target[1], bmab_model_path)


def install_pycocotools():
    url = 'https://github.com/Bing-su/dddetailer/releases/download/pycocotools/'
    files = {
        'Linux': {
            'x86_64': {
                '3.8': 'pycocotools-2.0.6-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
                '3.9': 'pycocotools-2.0.6-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
                '3.10': 'pycocotools-2.0.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
                '3.11': 'pycocotools-2.0.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
             }
        },
        'Windows': {
            'AMD64': {
                '3.8': 'pycocotools-2.0.6-cp38-cp38-win_amd64.whl',
                '3.9': 'pycocotools-2.0.6-cp39-cp39-win_amd64.whl',
                '3.10': 'pycocotools-2.0.6-cp310-cp310-win_amd64.whl',
                '3.11': 'pycocotools-2.0.6-cp311-cp311-win_amd64.whl',
            }
        }
    }

    python_version = platform.python_version_tuple()
    pkg_str = '%s.%s' % (python_version[0], python_version[1])
    system = platform.system()
    machine = platform.machine()
    file = files.get(system, {}).get(machine, {}).get(pkg_str)
    if file is None:
        print('Not found pycocotoos package', pkg_str, system, machine)
        return
    launch.run_pip('install %s' % (url + file))


def install_groundingdino():
    url = 'https://github.com/Bing-su/dddetailer/releases/download/pycocotools/'
    files = {
        'Linux': {
            'x86_64': {
                '3.10-2.0.1-11.7': 'groundingdino-0.1.0+torch2.0.1.cu117-cp310-cp310-linux_x86_64.whl',
                '3.10-2.0.1-11.8': 'groundingdino-0.1.0+torch2.0.1.cu118-cp310-cp310-linux_x86_64.whl',
                '3.11-2.0.1-11.7': 'groundingdino-0.1.0+torch2.0.1.cu117-cp311-cp311-linux_x86_64.whl',
                '3.11-2.0.1-11.8': 'groundingdino-0.1.0+torch2.0.1.cu118-cp311-cp311-linux_x86_64.whl',
            }
        },
        'Windows': {
            'AMD64': {
                '3.10-2.0.1-11.7': 'groundingdino-0.1.0+torch2.0.1.cu117-cp310-cp310-win_amd64.whl',
                '3.10-2.0.1-11.8': 'groundingdino-0.1.0+torch2.0.1.cu118-cp310-cp310-win_amd64.whl',
                '3.11-2.0.1-11.7': 'groundingdino-0.1.0+torch2.0.1.cu117-cp311-cp311-win_amd64.whl',
                '3.11-2.0.1-11.8': 'groundingdino-0.1.0+torch2.0.1.cu118-cp311-cp311-win_amd64.whl',
            }
        }
    }

    torch_version = parse(torch.__version__).base_version
    cuda_version = torch.version.cuda

    python_version = platform.python_version_tuple()
    pkg_str = '%s.%s-%s-%s' % (python_version[0], python_version[1], torch_version, cuda_version)
    system = platform.system()
    machine = platform.machine()
    file = files.get(system, {}).get(machine, {}).get(pkg_str)
    if file is None:
        print('Not found groudingdino package', pkg_str, system, machine)
        return
    launch.run_pip('install %s' % (url + file))


install_models()

required = {
    ('pycocotools', install_pycocotools),
    ('groundingdino', install_groundingdino)
}

for pack_name, func in required:
    if not launch.is_installed(pack_name):
        func()


