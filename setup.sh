#!/bin/bash
# Sim-to-Sim Policy Transfer Setup Script
# Target: Ubuntu 22.04 LTS
# Note: Tested on 22.04, may have issues on 24.04 due to Isaac Gym compatibility

set -e  # Exit on error

# Color definitions
NC='\033[0m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'

echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}  Sim-to-Sim Legged Locomotion Setup - VISTEC Exam${NC}"
echo -e "${BLUE}========================================================${NC}"

# Check Ubuntu version
if command -v lsb_release &> /dev/null; then
    UBUNTU_VER=$(lsb_release -rs)
    echo -e "${BLUE}Detected Ubuntu ${UBUNTU_VER}${NC}"
    if [[ "$UBUNTU_VER" != "22.04" ]]; then
        echo -e "${YELLOW}Warning: This script is tested on Ubuntu 22.04${NC}"
        echo -e "${YELLOW}You are running Ubuntu ${UBUNTU_VER}. Isaac Gym may have compatibility issues.${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Verify Isaac Gym exists
ISAACGYM_PATH="isaacgym"
if [ ! -d "$ISAACGYM_PATH" ]; then
    echo -e "${RED}Error: Isaac Gym not found at $ISAACGYM_PATH${NC}"
    echo -e "${YELLOW}Please download from: https://developer.nvidia.com/isaac-gym${NC}"
    echo -e "${YELLOW}Extract and place the 'isaacgym' folder inside ''${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Isaac Gym found at $ISAACGYM_PATH${NC}"

# Setup Conda environment
echo -e "${GREEN}[1/5] Setting up Conda environment 'unitree_rl'...${NC}"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo -e "${RED}Conda not found! Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

# Remove existing environment if it exists
if conda info --envs | grep -q "unitree_rl"; then
    echo -e "${YELLOW}Removing existing 'unitree_rl' environment...${NC}"
    conda env remove -n unitree_rl -y
fi

conda create -n unitree_rl python=3.8 -y
conda activate unitree_rl

# Install system-level Python 3.8 libraries (required for Isaac Gym)
echo -e "${GREEN}[2/5] Installing system dependencies for Isaac Gym...${NC}"
if [ ! -f "/usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0" ]; then
    echo -e "${YELLOW}Installing libpython3.8 (requires sudo password)...${NC}"
    
    # Add deadsnakes PPA for Python 3.8 on Ubuntu 22.04
    if ! apt-cache policy 2>/dev/null | grep -q "deadsnakes"; then
        echo -e "${BLUE}Adding deadsnakes PPA for Python 3.8...${NC}"
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
    fi
    
    # Install Python 3.8 libraries
    sudo apt-get install -y libpython3.8 libpython3.8-dev python3.8-dev
    
    # Verify installation
    if [ -f "/usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0" ]; then
        echo -e "${GREEN}✓ libpython3.8 installed successfully${NC}"
    else
        echo -e "${RED}Failed to install libpython3.8${NC}"
        echo -e "${YELLOW}Try manually: sudo apt-get install libpython3.8 libpython3.8-dev${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ libpython3.8 already installed${NC}"
fi

# Install Isaac Gym
echo -e "${GREEN}[3/5] Installing Isaac Gym Python interface...${NC}"
pip install -e "$ISAACGYM_PATH/python"

# Detect CUDA and install compatible PyTorch
echo -e "${GREEN}[4/5] Installing PyTorch and core dependencies...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found! Please install NVIDIA drivers.${NC}"
    exit 1
fi

CUDA_VER=$(nvidia-smi | grep -Po 'CUDA Version: \K(\d+\.\d+)' | head -n1)
CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
echo -e "${BLUE}Detected CUDA ${CUDA_VER}${NC}"

# PyTorch compatibility matrix
if [ "$CUDA_MAJOR" -ge 13 ]; then
    echo -e "${YELLOW}CUDA ${CUDA_VER} detected - using PyTorch 2.4.1${NC}"
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
elif (( $(echo "$CUDA_VER >= 12.1" | bc -l) )); then
    echo -e "${BLUE}Using PyTorch 2.1.0 for CUDA 12.1+${NC}"
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
elif (( $(echo "$CUDA_VER >= 11.8" | bc -l) )); then
    echo -e "${BLUE}Using PyTorch 2.0.1 for CUDA 11.8${NC}"
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${RED}CUDA version ${CUDA_VER} is too old. Minimum: 11.8${NC}"
    exit 1
fi

# Install remaining dependencies
echo -e "${BLUE}Installing MuJoCo, RL packages, and ML stack...${NC}"
pip install mujoco  # Native viewer included, no need for mujoco-viewer
pip install gymnasium tensorboard
pip install numpy scipy pandas scikit-learn
pip install matplotlib seaborn pillow
pip install typeguard pyyaml tqdm

# Install project packages
echo -e "${GREEN}[5/5] Installing project packages...${NC}"

# Install rsl-rl (check if it exists locally first)
if [ -d "rsl_rl" ]; then
    echo -e "${BLUE}Installing rsl-rl from local directory...${NC}"
    pip install -e rsl_rl/
elif [ -d "rsl_rl" ]; then
    echo -e "rsl_rl...${NC}"rsl_rl/
else
    echo -e "${BLUE}Installing rsl-rl from GitHub (v1.0.2)...${NC}"

# Install ActuatorNet (Only if it doesn't break the main env)
if [ -f "actuator_net/requirements.txt" ]; then
    echo -e "${BLUE}Attempting to install ActuatorNet dependencies...${NC}"
    # Use --no-deps or skip the heavy GUI stuff that causes the Qt error
    pip install matplotlib opencv-python python-dotenv || echo -e "${YELLOW}GUI dependencies failed, but core ML should work.${NC}"
else
    echo -e "${YELLOW}actuator_net/requirements.txt not found - skipping${NC}"
fi

# Verification
echo -e "${BLUE}========================================================${NC}"
echo -e "${GREEN}Installation complete! Running sanity checks...${NC}"
echo -e "${BLUE}========================================================${NC}"

PYTORCH_OK=false
ISAACGYM_OK=false
MUJOCO_OK=false

python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" && PYTORCH_OK=true || echo -e "${RED}✗ PyTorch FAILED${NC}"
python -c "from isaacgym import gymapi; print('✓ Isaac Gym OK')" && ISAACGYM_OK=true || echo -e "${RED}✗ Isaac Gym FAILED${NC}"
python -c "import mujoco; print(f'✓ MuJoCo {mujoco.__version__} OK')" && MUJOCO_OK=true || echo -e "${RED}✗ MuJoCo FAILED${NC}"

echo -e "${BLUE}========================================================${NC}"

if $PYTORCH_OK && $ISAACGYM_OK && $MUJOCO_OK; then
    echo -e "${GREEN}All checks passed! Setup successful.${NC}"
    echo -e "${YELLOW}Usage: conda activate unitree_rl${NC}"
else
    echo -e "${RED}Some checks failed. Please review errors above.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================================${NC}"