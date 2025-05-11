# FOR TRAIN
# export CUBLAS_WORKSPACE_CONFIG=":4096:8"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python src/utils/preprocess.py
python src/main.py --config configs/config.yaml

# FOR OPTIMIZATION
# python src/optimize.py --config configs/config.yaml