# PINN-MPC for VCC Evaporator Control

A PyTorch + MATLAB/Simulink implementation of a Physics-Informed Neural Network (PINN)-based Model Predictive Control (MPC) framework for vapor compression cycle (VCC) evaporators. 

By embedding first-principles ODEs derived from Moving Boundary models into neural networks, this approach delivers accurate and robust control without solving ODEs in real time applications.

Moving Boundary modeled ODEs are derived and written in casadi framework by Jisung Byun, alongside with MATLAB-Simulink simulation data for heat exchanger evaporators.

## How to use

```bash
# Install Python dependencies, and download data
bash scripts/setup.sh

# Train or test real time MPC. This requires real-time SIMULINK-pytorch MPC interface based on CSV file exchange. Contact author if needed.
bash scripts/dev.sh
```

## Results
![Modeling Results](dataset/assets/modeling_results.png)
![Optimization Results](dataset/assets/optimization_results.png)