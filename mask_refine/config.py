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
        
        print('====  Configuration Summary  ====')
        for k, v in self.options.items():
            print(f'{k:{left_width}}: {v}')
        print('=================================\n')
    
    def save(self, filename='./config.yaml'):
        with open(filename, 'w') as config_file:
            yaml.dump(self.options, config_file)


empty_mask_refine_config = Configuration({
    'dataset_path': None,
    'optical_flow_path': None,
    'mask_refine_path': None,
    'splits': [.80, .20],  # can be 0-2 elements
    'epochs_per_run':  100,
    'steps_per_epoch': 150,
    'debugging': True,
    'optical_flow_device': '/device:CPU:0',
    'model_device': '/device:CPU:0'
})
