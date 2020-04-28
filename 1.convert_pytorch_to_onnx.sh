set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate pytorch

MODEL_DIR="./model"

python export_pytorch_efficientnet_to_onnx.py \
-o $MODEL_DIR

#python optimize_onnx_model.py \
#-o $MODEL_DIR