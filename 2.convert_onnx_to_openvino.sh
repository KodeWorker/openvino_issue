set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate openvino

for MODEL_NAME in "efficientnet-b0" "efficientnet-b1" "efficientnet-b2" "efficientnet-b3" "efficientnet-b4" "efficientnet-b5" "efficientnet-b6" "efficientnet-b7"
do
	INPUT_MODEL="./model/"$MODEL_NAME".onnx"
	OUTPUT_DIR="./model/"$MODEL_NAME

	rm -r -f $OUTPUT_DIR
	mkdir $OUTPUT_DIR

	python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" \
	--input_shape=[1,3,224,224] \
	--data_type=FP16 \
	--input_model $INPUT_MODEL \
	--output_dir $OUTPUT_DIR
done