import yaml


def save_configuration(config: dict):
    with open('config.yaml', 'w') as config_file:
        yaml.dump(config, config_file)

def load_configuration(filename='config.yaml'):
    with open(filename, 'r') as config_file:
        yaml.

default_server_config = {
    'dataset_path': 'G:\\Team Drives\\COML-Fall-2018\\T0-VidSeg\\Data\\DAVIS',
    'optical_flow_path': './opt_flow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000',
    'splits': [.15],  # can be 0-2 elements
    'epochs_per_run':  100,
    'steps_per_epoch': 500,
    'debugging': True,
    'optical_flow_device': '/device:GPU:0',
    'model_device': '/device:GPU:1'
}

save_configuration(default_server_config)
