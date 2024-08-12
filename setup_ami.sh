#!/bin/bash

# Update and install dependencies
sudo apt-get update
sudo apt-get install -y build-essential curl

# Install CUDA and cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda

# Install Python packages
pip install tensorflow-gpu gym tqdm

# Clean up
rm cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
