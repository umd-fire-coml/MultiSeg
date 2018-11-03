#!usr/bin/env python

import argparse
import os
from os import path
from os import walk

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--string', dest='rm_str', type=str,
                    nargs=1, default=['mavish'])
parser.add_argument('-p', '--path', dest='start_path', type=str,
                    nargs=1, default=['.'])


args = parser.parse_args()

rm_str = args.rm_str[0]
start_path = args.start_path[0]

count = 0


def remove_mavish(root_dir):
    global count
    
    assert path.exists(root_dir)
    _, dirs, files = next(walk(root_dir))
    
    for f in files:
        if f == 'mavish_remove.py':
            continue
        if f.startswith(rm_str) or rm_str in f.split('.') or rm_str in f.split('-'):
            os.remove(path.join(root_dir, f))
            count += 1
    
    for sub_dir in dirs:
        remove_mavish(path.join(root_dir, sub_dir))


remove_mavish(start_path)
print(f'Removed {count} instances of \'{rm_str}\'-type files.')
