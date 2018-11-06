#!/bin/bash
sudo apt-get update
echo 'Y' | sudo apt-get install imagemagick php5-imagick
pip install -r requirements.txt