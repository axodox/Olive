import json
import os
from pathlib import Path
from olive.workflows import run as olive_run

script_dir = Path(os.path.dirname(os.path.abspath('__file__')))

olive_config = None
submodel_name = "unet_controlnet"
olice_config_path = script_dir / f"config_{submodel_name}.json"
olive_config_file = open(olice_config_path, "r")
olive_config = json.load(olive_config_file)

olive_run(olive_config)