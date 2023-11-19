import launch


def install_groudingdino():
	launch.run_pip('install pycocotools', 'sd-webui-bmab requirement: pycocotools')
	launch.run_pip('install git+https://github.com/IDEA-Research/GroundingDINO', 'sd-webui-bmab requirement: groundingdino')
