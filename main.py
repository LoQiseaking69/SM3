import boto3
import paramiko
import time
import yaml

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# AWS Configuration
instance_type = config['aws']['instance_type']
key_name = config['aws']['key_name']
security_group = config['aws']['security_group']
ami_id = config['aws']['ami_id']
num_instances = config['aws']['num_instances']
key_path = config['aws']['key_path']
script_path = config['aws']['script_path']

# Initialize AWS EC2 client
ec2 = boto3.client('ec2')

# Launch EC2 instances with GPUs
def launch_instances():
    print(f"Launching {num_instances} EC2 instances with type {instance_type}...")
    instances = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=num_instances,
        MaxCount=num_instances,
        KeyName=key_name,
        SecurityGroupIds=[security_group],
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'GPU-Instance'}]
        }],
    )
    instance_ids = [instance['InstanceId'] for instance in instances['Instances']]
    return instance_ids

# Wait for instances to initialize
def wait_for_instances(instance_ids):
    print("Waiting for instances to initialize...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=instance_ids)

    instances = ec2.describe_instances(InstanceIds=instance_ids)['Reservations'][0]['Instances']
    dns_names = [instance['PublicDnsName'] for instance in instances]
    return dns_names

# Set up each instance (install software, etc.)
def setup_instance(instance_dns):
    print(f"Setting up instance {instance_dns}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(instance_dns, username='ubuntu', key_filename=key_path)

    # Run setup commands (assuming the AMI is already configured with necessary software)
    commands = [
        "sudo apt-get update",
        "pip install tensorflow-gpu",
        "pip install gym tqdm"
    ]

    for command in commands:
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.recv_exit_status()  # Wait for command to complete

    ssh.close()

# Run sm3.py script on each instance
def run_script_on_instance(instance_dns):
    print(f"Running sm3.py on instance {instance_dns}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(instance_dns, username='ubuntu', key_filename=key_path)

    sftp = ssh.open_sftp()
    sftp.put(script_path, '/home/ubuntu/sm3.py')  # Upload sm3.py to the instance
    sftp.close()

    command = "python3 /home/ubuntu/sm3.py"
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout_text = stdout.read().decode('utf-8')
    stderr_text = stderr.read().decode('utf-8')

    ssh.close()
    print(f"Output from {instance_dns}:")
    print(stdout_text)
    if stderr_text:
        print(f"Errors from {instance_dns}:")
        print(stderr_text)

# Terminate EC2 instances
def terminate_instances(instance_ids):
    print(f"Terminating instances {instance_ids}...")
    ec2.terminate_instances(InstanceIds=instance_ids)

def main():
    # Launch EC2 instances
    instance_ids = launch_instances()

    # Wait for instances to be ready
    dns_names = wait_for_instances(instance_ids)
    print(f"Instances are ready: {dns_names}")

    # Set up each instance
    for dns in dns_names:
        setup_instance(dns)

    # Run sm3.py on each instance
    for dns in dns_names:
        run_script_on_instance(dns)

    # Optionally, terminate instances after training
    terminate_instances(instance_ids)

if __name__ == "__main__":
    main()
