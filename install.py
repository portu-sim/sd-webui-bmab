import platform

import launch
import torch

from packaging.version import parse


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
    url = 'https://github.com/Bing-su/GroundingDINO/releases/download/'
    files = {
        'Linux': {
            'x86_64': {
                '3.10-1.13.1-11.7': 'wheel-0.1.0/groundingdino-0.1.0+torch1.13.1.cu117-cp310-cp310-linux_x86_64.whl',
                '3.9-1.13.1-11.7': 'wheel-0.1.0/groundingdino-0.1.0+torch1.13.1.cu117-cp39-cp39-linux_x86_64.whl',
                '3.10-2.0.1-11.7': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu117-cp310-cp310-linux_x86_64.whl',
                '3.10-2.0.1-11.8': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu118-cp310-cp310-linux_x86_64.whl',
                '3.11-2.0.1-11.7': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu117-cp311-cp311-linux_x86_64.whl',
                '3.11-2.0.1-11.8': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu118-cp311-cp311-linux_x86_64.whl',
            }
        },
        'Windows': {
            'AMD64': {
                '3.10-2.0.1-11.7': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu117-cp310-cp310-win_amd64.whl',
                '3.10-2.0.1-11.8': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu118-cp310-cp310-win_amd64.whl',
                '3.11-2.0.1-11.7': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu117-cp311-cp311-win_amd64.whl',
                '3.11-2.0.1-11.8': '0.1.0-2.0.1/groundingdino-0.1.0+torch2.0.1.cu118-cp311-cp311-win_amd64.whl',
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
    print('install from', file)
    if file is None:
        if system == 'Linux':
            print('install from gitgub')
            launch.run_pip('install git+https://github.com/IDEA-Research/GroundingDINO', 'sd-webui-bmab requirement: groundingdino')
            return
        print('Not found groudingdino package', pkg_str, system, machine)
        return
    launch.run_pip('install %s' % (url + file), 'sd-webui-bmab requirement: groundingdino')


def install_segmentanything():
    launch.run_pip('install segment_anything')


def install_segmentanything_hq():
    launch.run_pip('install segment-anything-hq')


def install_ultralytics():
    launch.run_pip('install ultralytics')


required = {
    ('pycocotools', install_pycocotools),
    ('groundingdino', install_groundingdino),
    ('segment_anything', install_segmentanything),
    ('segment_anything_hq', install_segmentanything_hq),
    ('ultralytics', install_ultralytics)
}

for pack_name, func in required:
    if not launch.is_installed(pack_name):
        func()


