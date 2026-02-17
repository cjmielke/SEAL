"""
Util functions for general experiments
"""
import getpass
from pathlib import Path
import os
import yaml


def update_config(args):
    """
    Override config file with CLI arguments

    Outputs
    =======
    conf: dictionary of parameters
    """
    conf = load_config(args)
    conf_args = vars(args)
    
    if conf is None:
        conf = {}
    for k, v in conf_args.items():
        if v is not None:
            set_nested(conf, k, v)
    return conf

def load_config(args): 

    if args.config.split('.')[-1] == 'yaml':
        with open(args.config, 'r') as f:
            main_conf = yaml.safe_load(f)
    else:
        raise NotImplementedError("Config file not supplied")
    
    # config_dir = Path(os.path.dirname(args.config))
    user_config_dir = Path("conf/user/")
    # load user config
    user = getpass.getuser()
    # try: 
    user_config_path = user_config_dir / f'{user}.yaml'
    if os.path.exists(user_config_path):
        print(f"Loading user config: {user_config_path}")
        with open(user_config_path, 'r') as f:
            user_conf = yaml.safe_load(f)
        # append to main_conf
        for key, value in user_conf.items():
            main_conf[key] = value
    
    return main_conf

def set_nested(conf, key_path, value):
    keys = key_path.split(".")
    d = conf
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value