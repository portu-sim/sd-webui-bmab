import launch


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


