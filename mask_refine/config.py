import yaml
from abc import ABC

from typing import Any, Dict, Union


class Configuration(ABC):
    def __init__(self, options: Union[Dict[str, Any], str] = './config.yaml'):
        if isinstance(options, str):
            with open(options, 'r') as config_file:
                self.options = yaml.load(config_file)
        else:
            self.options = options
        
    def __getattr__(self, item):
        return self.options[item]
    
    def summary(self):
        left_width = 1 + max(map(len, self.options.keys()))
        
        print('== Configuration Summary ==')
        for k, v in self.options.items():
            print(f'{k:{left_width}}: {v}')
    
    def save(self, filename='./config.yaml'):
        with open(filename, 'w') as config_file:
            yaml.dump(self.options, config_file)


default_server_config = Configuration({
    'dataset_path': 'G:\\Team Drives\\COML-Fall-2018\\T0-VidSeg\\Data\\DAVIS',
    'optical_flow_path': './opt_flow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000',
    'mask_refine_path': None,  # TODO find out this path
    'splits': [.85, .15],  # can be 0-2 elements
    'epochs_per_run':  100,
    'steps_per_epoch': 500,
    'debugging': True,
    'optical_flow_device': '/device:GPU:0',
    'model_device': '/device:GPU:1'
})

default_server_config.save()
