# scripts/user-data.sh
#!/bin/bash

# Update system
yum update -y

# Install Docker
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Install Python and pip
yum install -y python3 python3-pip

# Clone repository and deploy
cd /home/ec2-user
git clone https://github.com/your-username/mlops-pipeline.git
cd mlops-pipeline

# Build and run Docker container
docker build -t ml-model-api .
docker run -d -p 80:5000 --name ml-api ml-model-api

echo "ML API deployed successfully"
