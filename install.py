import launch
import os
import glob
import requests
from modules.paths import models_path
#from basicsr.utils.download_util import load_file_from_url
#from sd_bmab.util import lazy_loader
from huggingface_hub import hf_hub_download
from urllib.parse import urlparse


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


# Define the targets dictionary outside the lazy_loader function
targets_huggingface = {
    'groundingdino_swint_ogc.pth': 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth',
    'face_yolov8n.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt',
    'face_yolov8n_v2.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt',
    'face_yolov8m.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt',
    'face_yolov8s.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt',
    'hand_yolov8n.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt',
    'hand_yolov8s.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt',
    'person_yolov8m-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt',
    'person_yolov8n-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt',
    'person_yolov8s-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt',
    'sam_hq_vit_b.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth',
    'sam_hq_vit_h.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth',
    'sam_hq_vit_l.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth',
    'sam_hq_vit_tiny.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth',
    'bmab_face_nm_yolov8n.pt': 'https://huggingface.co/portu-sim/bmab/resolve/main/bmab_face_nm_yolov8n.pt',
    'bmab_face_sm_yolov8n.pt': 'https://huggingface.co/portu-sim/bmab/resolve/main/bmab_face_sm_yolov8n.pt',
    'bmab_hand_yolov8n.pt': 'https://huggingface.co/portu-sim/bmab/resolve/main/bmab_hand_yolov8n.pt',
    'ControlNetLama.pth': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth'
}

for filename, url in targets_huggingface.items():
    # Extracting repo_id from the URL
    repo_id = url.split('/')[3] + '/' + url.split('/')[4]

    bmab_model_path = os.path.join(models_path, "bmab")
    files = glob.glob(bmab_model_path)

    if filename not in files:
        # Downloading the file with destination folder specified
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=bmab_model_path)
        print(f"Downloaded {filename} to {bmab_model_path}")


targets_no-huggingface = {
    'sam_vit_b_01ec64.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'sam_vit_l_0b3195.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'GroundingDINO_SwinT_OGC.py': 'https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py'
}
