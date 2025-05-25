import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import font_manager

# Load Times New Roman font
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
font_prop = font_manager.FontProperties(fname=font_path, size=11)

def plot_prediction(lstm_path, gru_path, rnn_path, ans_path, start=0, window=500):
    # Load CSVs
    lstm = pd.read_csv(lstm_path).to_numpy()
    gru = pd.read_csv(gru_path).to_numpy()
    rnn = pd.read_csv(rnn_path).to_numpy()
    ans = pd.read_csv(ans_path).to_numpy()

    ans = ans[70:, 1:3]
    rnn = rnn[40:, 0:2]

    # Trim to minimal length
    min_len = min(len(lstm), len(gru), len(rnn), len(ans))
    print(len(lstm), len(gru), len(rnn), len(ans), min_len)
    end = min(start + window, min_len)
    time = np.arange(start, end) * 2

    # Extract slices
    p_lstm, h_lstm = lstm[start:end, 0], lstm[start:end, 1]
    p_ans, h_ans = ans[start:end, 0], ans[start:end, 1]
    p_gru, h_gru = gru[start:end, 0], gru[start:end, 1]
    p_rnn, h_rnn = rnn[start:end, 0], rnn[start:end, 1]

    # Pressure plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, p_lstm, 'g-', label='LSTM')
    ax.plot(time, p_ans, 'k-', label='Answer')
    ax.plot(time, p_gru, 'b-', label='GRU')
    ax.plot(time, p_rnn, 'r-', label='RNN')

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

    # Enthalpy plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time, h_lstm, 'g-', label='LSTM')
    ax.plot(time, h_ans, 'k-', label='Answer')
    ax.plot(time, h_gru, 'b-', label='GRU')
    ax.plot(time, h_rnn, 'r-', label='RNN')

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

plot_prediction("LSTM.csv", "GRU.csv", "RNN.csv", "dataset.csv", start=42800, window=200)