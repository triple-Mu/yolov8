import os
import subprocess
from datetime import datetime
from pathlib import Path

log = Path('.log')
now = datetime.now()
formatted_time = now.strftime('%Y_%m_%d')
log = log / formatted_time

log.mkdir(parents=True, exist_ok=True)

cmd = f'yolo detect train ' \
      f'model=yolov8x-p6.yaml ' \
      f'data={{}} ' \
      f'epochs=600 ' \
      f'imgsz=1600 ' \
      f'batch=32 ' \
      f'device=0,1,2,3 ' \
      f'project=training/kfold ' \
      f'name=exp{{}} ' \
      f'pretrained=False ' \
      f'seed=43' # noqa F541

sed = f'sed ' \
      fr"'11s/.*/path: \/dev\/shm\/coco2yolov5\/exp{{}}/g' " \
      f'coco-self.yaml ' \
      f'> {{}}' # noqa F541

for i in range(1, 11):
    now = datetime.now()
    formatted_time = now.strftime('%Y_%m_%d_%H_%M')
    log_file = log / f'{formatted_time}_exp{i}.log'
    yaml_file = log / f'exp{i}.yaml'
    _sed = sed.format(i, yaml_file)
    _cmd = cmd.format(yaml_file, i, i)
    _string = _sed + '\n\n' + _cmd
    log_file.write_text(_string.replace(' ', ' \\\n'))
    print(_sed)
    subprocess.run(_sed, shell=True, check=True, env=os.environ)
    print(_cmd)
    subprocess.run(_cmd, shell=True, check=True, env=os.environ)
