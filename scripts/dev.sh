# python src/utils/preprocess.py
# python src/test.py --config configs/config.yaml

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python src/utils/preprocess.py
python src/main.py --config configs/config.yaml

# python src/optimize.py --config configs/config.yaml