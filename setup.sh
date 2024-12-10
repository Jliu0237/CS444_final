#!/bin/bash

# Function to check CUDA installation
check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA not found. Please install CUDA first."
        echo "Installation instructions:"
        echo "1. Visit: https://developer.nvidia.com/cuda-downloads"
        echo "2. Select your operating system and follow the installation guide"
        echo "3. For Ubuntu, you can use the following commands:"
        echo ""
        echo "# Add NVIDIA package repositories"
        echo "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"
        echo "sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
        echo "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub"
        echo "sudo add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /\""
        echo ""
        echo "# Install CUDA"
        echo "sudo apt update"
        echo "sudo apt install cuda"
        echo ""
        echo "# Add CUDA to PATH (add to ~/.bashrc)"
        echo "export PATH=/usr/local/cuda/bin:\${PATH}"
        echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}"
        echo ""
        echo "After installation, reboot your system and run this script again."
        exit 1
    fi
}

# Function to check GPU availability
check_gpu() {
    if ! nvidia-smi &> /dev/null; then
        echo "No NVIDIA GPU detected or driver not installed."
        echo "Please make sure you have an NVIDIA GPU and install the drivers:"
        echo ""
        echo "For Ubuntu:"
        echo "sudo ubuntu-drivers devices"
        echo "sudo ubuntu-drivers autoinstall"
        echo "sudo reboot"
        exit 1
    fi
}

# Check CUDA and GPU
echo "Checking CUDA installation..."
check_cuda

echo "Checking GPU availability..."
check_gpu

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "Detected CUDA version: $CUDA_VERSION"

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support based on detected version
echo "Installing PyTorch with CUDA $CUDA_VERSION support..."
if [[ $CUDA_VERSION == 11.* ]]; then
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
elif [[ $CUDA_VERSION == 12.* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Unsupported CUDA version. Please check PyTorch documentation for compatibility."
    exit 1
fi

# Install detectron2
echo "Installing detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/cache
mkdir -p output

echo "Setup completed successfully!"
echo "You can now run the training script with:"
echo "python train_net.py --data-dir ./data --max-images 1000 --max-iterations 5000"