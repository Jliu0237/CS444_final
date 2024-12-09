# System Update and Dependencies
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential python3-dev git wget

# Create and Activate Conda Environment
conda create -n cs444_final python=3.8 -y
conda activate cs444_final

# Install PyTorch and Core Libraries
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install All Required Python Packages
pip install opencv-python \
    pandas \
    numpy \
    matplotlib \
    pillow \
    scikit-learn \
    albumentations \
    jupyter \
    notebook \
    pycocotools \
    seaborn \
    tensorboard \
    segmentation-models-pytorch \
    efficientnet-pytorch \
    mlflow \
    wandb \
    black \
    flake8 \
    pytest \
    kaggle \
    tqdm \
    imageio \
    imgaug \
    joblib

# Create Data Directory Structure
mkdir -p data
cd data

# Create Kaggle Directory (you'll need to manually add kaggle.json later)
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

# Verify Installation
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import detectron2; print('Detectron2 Available')"
python -c "import albumentations; print('Albumentations Version:', albumentations.__version__)"

echo "Installation complete! Remember to:"
echo "1. Add your kaggle.json to ~/.kaggle/"
echo "2. Download the specific competition dataset using: kaggle competitions download -c [competition-name]"
echo "3. Download the Fashionpedia dataset from the official source"