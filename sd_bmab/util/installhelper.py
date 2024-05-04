import launch
import torch
import platform


groundingdino_install_info = {
	'2.1.2+cu121:3.10:linux:amd64': 'https://github.com/Bing-su/GroundingDINO/releases/download/v23.9.27/groundingdino-23.9.27+torch2.1.0.cu121-cp310-cp310-manylinux_2_34_x86_64.whl',
	'2.1.2+cu121:3.10:windows:amd64': 'https://github.com/Bing-su/GroundingDINO/releases/download/v23.9.27/groundingdino-23.9.27+torch2.1.0.cu121-cp310-cp310-win_amd64.whl',
	'2.1.2+cu121:3.11:linux:amd64': 'https://github.com/Bing-su/GroundingDINO/releases/download/v23.9.27/groundingdino-23.9.27+torch2.1.0.cu121-cp311-cp311-manylinux_2_34_x86_64.whl',
	'2.1.2+cu121:3.11:windows:amd64': 'https://github.com/Bing-su/GroundingDINO/releases/download/v23.9.27/groundingdino-23.9.27+torch2.1.0.cu121-cp311-cp311-win_amd64.whl',
	'2.1.2+cu121:3.9:linux:amd64': 'https://github.com/Bing-su/GroundingDINO/releases/download/v23.9.27/groundingdino-23.9.27+torch2.1.0.cu121-cp39-cp39-manylinux_2_34_x86_64.whl',
	'2.1.2+cu121:3.9:windows:amd64': 'https://github.com/Bing-su/GroundingDINO/releases/download/v23.9.27/groundingdino-23.9.27+torch2.1.0.cu121-cp39-cp39-win_amd64.whl',
}

groundingdino_install_replacement = {
	'2.1.0+cu121:3.10:linux:amd64': '2.1.2+cu121:3.10:linux:amd64',
	'2.1.1+cu121:3.10:linux:amd64': '2.1.2+cu121:3.10:linux:amd64',
}


available_packages = ['GroundingDINO']


def install_groudingdino(url):
	print('Install pycocotools')
	launch.run_pip('install pycocotools', 'sd-webui-bmab requirement: pycocotools')

	print(f'Install groundingdino from {url}. It takes minutes.')
	launch.run_pip(f'install {url}', 'sd-webui-bmab requirement: groundingdino')
	print('Done.')


def get_condition():
	torch_version = torch.__version__
	pv = platform.python_version_tuple()
	system = 'windows' if platform.system() == 'Windows' else 'linux'
	machine = 'amd64' if platform.machine() == 'AMD64' else 'x86_64'
	return f'{torch_version}:{pv[0]}.{pv[1]}:{system}:{machine}'


def install(pkg_name, dd_pkg, markdown_install):
	groundingdino_cuda_name = 'GroundingDINO for CUDA'
	groundingdino_selected = pkg_name

	def add_new_available(cond):
		msg = f'GroundingDINO for CUDA Found {cond}. Please select GroudingDINO or {groundingdino_cuda_name}'
		newname = f'{groundingdino_cuda_name}-{cond}'
		if newname not in available_packages:
			available_packages.append(newname)
			print(available_packages)
		return msg, newname

	if pkg_name == 'GroundingDINO':
		cond = get_condition()
		replacement = groundingdino_install_replacement.get(cond)
		if replacement is not None and groundingdino_cuda_name not in available_packages:
			msg, groundingdino_selected = add_new_available(cond)
		else:
			groudingdino_for_cuda = groundingdino_install_info.get(cond)
			if groudingdino_for_cuda is None:
				url = 'git+https://github.com/IDEA-Research/GroundingDINO'
				install_groudingdino(url)
				msg = f'Nothing found for this cuda system {cond}. Software version of GroundingDINO installed (VERY SLOW!!!)'
			else:
				msg, groundingdino_selected = add_new_available(cond)
	elif pkg_name.startswith(groundingdino_cuda_name):
		groudingdino_for_cuda_cond = pkg_name[len(groundingdino_cuda_name)+1:]
		groudingdino_for_cuda = groundingdino_install_info.get(groudingdino_for_cuda_cond)
		if groudingdino_for_cuda is not None:
			install_groudingdino(groudingdino_for_cuda)
			msg = f'{groundingdino_cuda_name} installed. {groudingdino_for_cuda}'
			groundingdino_selected = f'{groundingdino_cuda_name}-{groudingdino_for_cuda_cond}'
		else:
			msg = 'Nothing installed.'
	else:
		msg = 'Nothing installed.'

	return {
		dd_pkg: {
			'choices': available_packages,
			'value': groundingdino_selected,
			'__type__': 'update'
		},
		markdown_install: {
			'value': msg,
			'__type__': 'update'
		}
	}

