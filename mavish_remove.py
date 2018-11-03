#!usr/bin/env python

import os
from os import path
from os import walk


def remove_mavish(root_dir):
    assert path.exists(root_dir)
    _, dirs, files = next(walk(root_dir))
    
    if 'mavish' in files:
        os.remove(path.join(root_dir, 'mavish'))
    
    for sub_dir in dirs:
        remove_mavish(path.join(root_dir, sub_dir))


remove_mavish('.')

