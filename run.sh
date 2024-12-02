#!/bin/bash
config_path=./configs/AFHQv2.yaml
echo ${config_path}
python train.py --config ${config_path}