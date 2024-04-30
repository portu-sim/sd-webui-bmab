import os
import shutil
import sd_bmab


def check_directory():
	target = ['cache', 'ipadapter', 'pose', 'saved']
	bmab_path = os.path.dirname(sd_bmab.__file__)
	dest_path = os.path.normpath(os.path.join(bmab_path, f'../resources'))
	for t in target:
		path = os.path.normpath(os.path.join(bmab_path, f'../{t}'))
		if os.path.exists(path) and os.path.isdir(path):
			shutil.move(path, dest_path)
