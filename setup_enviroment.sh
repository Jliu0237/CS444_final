# System Update and Dependencies
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential python3-dev git wget unzip

# Create and Activate Conda Environment
conda create -n fashion python=3.8 -y
conda activate fashion

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

# Create Kaggle Directory
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

# Download iMaterialist Fashion Dataset
kaggle competitions download -c imaterialist-fashion-2019-FGVC6
unzip imaterialist-fashion-2019-FGVC6.zip

# Verify Installation
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import detectron2; print('Detectron2 Available')"
python -c "import albumentations; print('Albumentations Version:', albumentations.__version__)"

echo "Installation complete! Important next steps:"
echo "1. Before running the kaggle download command, make sure to:"
echo "   - Create a Kaggle account if you haven't"
echo "   - Go to kaggle.com -> Your Account -> Create New API Token"
echo "   - Download the kaggle.json file"
echo "   - Move it to ~/.kaggle/kaggle.json"
echo "2. The dataset includes:"
echo "   - train.csv: training annotations"
echo "   - train/: directory containing training images"
echo "   - test/: directory containing test images"
echo "   - sample_submission.csv: sample submission file"