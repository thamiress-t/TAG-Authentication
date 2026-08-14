#!/bin/bash
# 
# start_jupyter_gpu.sh
# Helper script to start Jupyter with TensorFlow GPU support in WSL
#
# Usage:
#   chmod +x start_jupyter_gpu.sh
#   ./start_jupyter_gpu.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  Jupyter with TensorFlow GPU Support (WSL)          ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

# Check if venv exists
VENV_PATH="/home/thami/venv_tf_gpu"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found venv at: $VENV_PATH${NC}"

# Activate venv
echo -e "${YELLOW}🔄 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Check if TensorFlow is installed
if ! python -c "import tensorflow" 2>/dev/null; then
    echo -e "${RED}❌ Error: TensorFlow not found in venv${NC}"
    exit 1
fi

# Get TensorFlow version
TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)")
echo -e "${GREEN}✅ TensorFlow: $TF_VERSION${NC}"

# Check for GPU
GPU_COUNT=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")
echo -e "${GREEN}✅ GPUs detected: $GPU_COUNT${NC}"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  WARNING: No GPU detected - training will use CPU (slower)${NC}"
else
    echo -e "${GREEN}✅ GPU will be used for training${NC}"
fi

# Ask which Jupyter to use
echo -e "${BLUE}Choose Jupyter interface:${NC}"
echo "  1) Jupyter Lab (recommended, modern)"
echo "  2) Jupyter Notebook (classic)"
echo -e "${YELLOW}Enter choice (1 or 2):${NC} "
read -r CHOICE

case $CHOICE in
    1)
        echo -e "${YELLOW}🚀 Starting Jupyter Lab...${NC}"
        jupyter lab
        ;;
    2)
        echo -e "${YELLOW}🚀 Starting Jupyter Notebook...${NC}"
        jupyter notebook
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac
