import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import font_manager

# Load Times New Roman font
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
font_prop = font_manager.FontProperties(fname=font_path, size=11)


def plot_prediction(bb_path, hyb_path, pinn_path, ans_path, start=0, window=500):
    # Load CSVs
    bb = pd.read_csv(bb_path).to_numpy()
    hyb = pd.read_csv(hyb_path).to_numpy()
    pinn = pd.read_csv(pinn_path).to_numpy()
    ans = pd.read_csv(ans_path).to_numpy()
    ans = ans[70:, 1:3]
    pinn = pinn[40:, 0:2]

    # Trim to minimal length
    min_len = min(len(bb), len(hyb), len(pinn), len(ans))
    print(len(bb), len(hyb), len(pinn), len(ans), min_len)
    end = min(start + window, min_len)
    time = np.arange(start, end) * 2

    # Extract slices
    p_ans, h_ans = ans[start:end, 0], ans[start:end, 1]
    p_bb, h_bb = bb[start:end, 0], bb[start:end, 1]
    p_hyb, h_hyb = hyb[start:end, 0], hyb[start:end, 1]
    p_pinn, h_pinn = pinn[start:end, 0], pinn[start:end, 1]

    # --- Pressure Plot ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, p_ans, 'k-', label='Simulation')
    ax.plot(time, p_pinn, 'r-', label='PINN')
    ax.plot(time, p_hyb, 'b-', label='Hybrid')
    ax.plot(time, p_bb, 'g-', label='Black-box')

    ax.set_xlabel("Time (sec)", fontproperties=font_prop)
    ax.set_ylabel(r"$p$ (kPa)", fontproperties=font_prop)
    ax.set_title("Pressure, Validation", fontproperties=font_prop)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    fig.savefig("plot_p_val.png", dpi=300)
    plt.close()

    # --- Enthalpy Plot ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, h_ans, 'k-', label='Simulation')
    ax.plot(time, h_pinn, 'r-', label='PINN')
    ax.plot(time, h_hyb, 'b-', label='Hybrid')
    ax.plot(time, h_bb, 'g-', label='Black-box')

    ax.set_xlabel("Time (sec)", fontproperties=font_prop)
    ax.set_ylabel(r"$h_{out}$ (kJ/kg)", fontproperties=font_prop)
    ax.set_title("Enthalpy, Validation", fontproperties=font_prop)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    ax.grid(True, linewidth=0.5)
    ax.legend(prop=font_prop, fontsize=10)
    fig.tight_layout()
    fig.savefig("plot_h_val.png", dpi=300)
    plt.close()
plot_prediction("savefiles/y_traj_bb_val.csv", "savefiles/y_traj_hyb_val.csv", "savefiles/pinn_total.csv", "savefiles/dataset.csv", start=42800, window=200)

"""
def plot_convergence(csv_path):
    data = pd.read_csv(csv_path)

    time = data["time"].to_numpy()
    ratio = data["noise to time ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, ratio, 'k-', label='Simulation')

    ax.set_xlabel("Convergence time(sec)", fontproperties=font_prop)
    ax.set_ylabel("Noise/time ratio", fontproperties=font_prop)
    ax.set_title("Noise to Convergence time ratio", fontproperties=font_prop)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    fig.tight_layout()
    fig.savefig("NtoT.png", dpi=300)
    plt.close()
plot_convergence("savefiles/NTS.csv")


def plot_T(csv_path):
    data = pd.read_csv(csv_path)

    time = data["time"].to_numpy()
    value = data["temperature"].to_numpy()
    value2 = data["temperature2"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, value, 'k-', label='Simulated T')
    ax.plot(time, value2, 'r-', label='predicted T')

    ax.set_xlabel("Time(sec)", fontproperties=font_prop)
    ax.set_ylabel("Temperature(°C)", fontproperties=font_prop)
    ax.set_title("Utility Coolant Outlet Temperature", fontproperties=font_prop)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    fig.tight_layout()
    ax.legend(prop=font_prop, fontsize=10)
    fig.savefig("T_cool.png", dpi=300)
    plt.close()
plot_T("savefiles/T_cool.csv")
"""
def plot_time(csv_path):
    data = pd.read_csv(csv_path)

    time = data["time"].to_numpy()
    value = data["zeta"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, value, 'k-', label='Simulation')

    ax.set_xlabel("Time", fontproperties=font_prop)
    ax.set_ylabel("Zeta", fontproperties=font_prop)
    ax.set_title("Zeta", fontproperties=font_prop)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    fig.tight_layout()
    fig.savefig("zeta.png", dpi=300)
    plt.close()
    
plot_time("savefiles/zeta.csv")
"""