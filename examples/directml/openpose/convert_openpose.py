# Download model weights from here: https://github.com/isl-org/MiDaS

import json
import os
import pathlib
from olive.workflows import run as olive_run

script_dir = pathlib.Path(__file__).parent.resolve()

olive_config = None
olive_config_path = script_dir / f"config_openpose.json"
olive_config_file = open(olive_config_path, "r")
olive_config = json.load(olive_config_file)

olive_run(olive_config)