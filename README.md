# SM3 Pipeline

## Overview

The SM3 Pipeline is an automated system designed for training a Q-learning agent using cloud-based infrastructure on AWS EC2 instances equipped with GPUs. The pipeline aims to  encompass everything from infrastructure provisioning, model building, and training to cleanup, allowing for efficient and scalable machine learning model development.

The core model integrates advanced neural network layers, including Restricted Boltzmann Machines (RBMs) and Multi-Head Attention mechanisms, to handle complex tasks like controlling a humanoid in the Humanoid-v3 environment.

## Components

### 1. `main.py`
The `main.py` script automates the entire lifecycle of EC2 instances for model training. This includes:

- **Launching EC2 instances** with GPU capabilities.
- **Setting up the environment** on each instance by installing necessary dependencies.
- **Executing the model training script** (`sm3.py`).
- **Terminating the instances** after training is complete.

### 2. `sm3.py`
This is the core model training script that:

- Implements a Q-learning agent with a sophisticated neural network architecture.
- Uses Prioritized Experience Replay, RBM layers, and Multi-Head Attention to improve learning.
- Trains the agent in the Humanoid-v3 environment and evaluates its performance over multiple episodes.

### 3. `config.yaml`
The `config.yaml` file centralizes all configuration parameters, including AWS settings and training parameters. This setup allows for easy adjustments without modifying the main scripts.

### 4. `setup_ami.sh`
This script is used to prepare a custom Amazon Machine Image (AMI) with all the necessary software pre-installed, such as CUDA, cuDNN, and TensorFlow. Using a pre-configured AMI significantly reduces the time needed to set up each EC2 instance.

## Getting Started

### Prerequisites
- **AWS Account**: Ensure you have an AWS account with permissions to create and manage EC2 instances.
- **AWS CLI**: Install and configure the AWS CLI on your local machine to interact with AWS services.
- **Python 3.x**: Install Python 3.x with the following packages:
  - `boto3`: AWS SDK for Python.
  - `paramiko`: For SSH connections.
  - `PyYAML`: For parsing the configuration file.
- **SSH Key Pair**: Ensure you have an SSH key pair configured for accessing EC2 instances. This key pair must be added to your AWS account.

### Setup and Execution

1. **Prepare a Custom AMI** (Optional):
   - Run the `setup_ami.sh` script to create a custom AMI with all necessary dependencies. This step is optional but recommended for faster instance setup.

2. **Update Configuration**:
   - Edit the `config.yaml` file to match your specific AWS setup and training parameters. This includes instance types, key names, security group IDs, and model training settings.

3. **Run the Pipeline**:
   - Execute `main.py` to initiate the entire pipeline. This script will automatically handle the infrastructure setup, training, and cleanup.

### Cleanup

After training is complete, `main.py` will automatically terminate the EC2 instances to prevent unnecessary costs. However, it's recommended to double-check your AWS resources to ensure that no unwanted resources are left running.

```
SM3_Pipeline/
│
├── main.py                 # Automates EC2 instance lifecycle and training
├── sm3.py                  # Core model training script
├── config.yaml             # Configuration file for AWS and training settings
├── setup_ami.sh            # Script to prepare a custom AMI (optional)
└── README.md               # Detailed documentation of the project
```
## Usage Notes

- **Checkpointing**: The `sm3.py` script saves model checkpoints every 100 episodes, allowing you to resume training or evaluate a model without starting from scratch.
- **Logging**: Detailed logging is provided throughout the process to help track progress and diagnose issues.
- **Model Evaluation**: After training, the model is evaluated over a specified number of episodes to determine its performance.

## License

This project is licensed under the BSD 3-Clause License, meaning you are free to use, modify, and distribute the software as long as the original license terms are retained.
