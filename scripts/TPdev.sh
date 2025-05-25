# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# FOR TRAIN
python src/utils/preprocess.py
python src/main.py --config configs/TPconfig.yaml
