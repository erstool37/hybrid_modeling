export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# FOR TRAIN
# python src/utils/preprocess.py
# python src/main.py --config configs/config.yaml

# FOR OPTIMIZATION
# python src/optimize.py --config configs/config.yaml
python src/optimize_hybrid.py --config configs/configHybrid.yaml