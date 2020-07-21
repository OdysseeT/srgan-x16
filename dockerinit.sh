#!/bin/sh


apt update
apt install -y libsm6 libxext6 libxrender-dev git

pip install --upgrade pip
pip install -r requirements-tf2.txt
