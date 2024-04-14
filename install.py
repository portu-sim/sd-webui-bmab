import launch
import os
import glob
import requests
from modules.paths import models_path
from sd_bmab.util import lazy_loader


def install_segmentanything():
    launch.run_pip('install segment_anything')


def install_segmentanything_hq():
    launch.run_pip('install segment_anything_hq')


def install_ultralytics():
    launch.run_pip('install ultralytics')


def install_basicsr():
    launch.run_pip('install basicsr')


required = {
    ('segment_anything', install_segmentanything),
    ('segment_anything_hq', install_segmentanything_hq),
    ('ultralytics', install_ultralytics),
    ('basicsr', install_basicsr)
}

for pack_name, func in required:
    if not launch.is_installed(pack_name):
        func()


def load_file_from_url(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Load all files listed in the targets dictionary
loaded_files = {}
for filename in targets.keys():
    loaded_files[filename] = lazy_loader(filename)

# Now you can use the loaded file paths as needed
for filename, file_path in loaded_files.items():
    print(f"Loaded {filename} from {file_path}")

