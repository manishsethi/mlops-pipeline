# scripts/deploy-localstack.sh
#!/bin/bash

# Install LocalStack
pip install localstack awscli-local

# Start LocalStack
localstack start -d

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to be ready..."
sleep 10

# Create EC2 instance
awslocal ec2 run-instances \
    --image-id ami-12345678 \
    --count 1 \
    --instance-type t2.micro \
    --key-name my-key \
    --user-data file://scripts/user-data.sh

echo "LocalStack EC2 instance created for testing"
