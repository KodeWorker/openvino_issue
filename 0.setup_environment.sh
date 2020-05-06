set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"

OPENVINO_DIR="C:\Program Files (x86)\IntelSWTools\openvino"

# Create PyTorch environment
conda create --name --yes pytorch python=3.6
conda activate pytorch
pip install --yes -r pytorch_requirements.txt
conda deactivate
# Create OpenVINO environment
conda create --name --yes openvino python=3.6
conda activate openvino
pip install --yes -r openvino_requirements.txt
conda deactivate

