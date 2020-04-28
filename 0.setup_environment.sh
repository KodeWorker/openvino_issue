set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"

OPENVINO_DIR="C:\Program Files (x86)\IntelSWTools\openvino"

# Create PyTorch environment
conda create --name pytorch python=3.6
pip install -r pytorch_requirements.txt
conda deactivate
# Create OpenVINO environment
conda create --name openvino python=3.6
pip install -r openvino_requirements.txt
conda deactivate

