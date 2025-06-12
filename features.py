import numpy as np
import pandas as pd
from scipy import signal
from numpy import trapz
import streamlit as st
def calculate_psd(data, sfreq):
nperseg = len(data)
return signal.welch(data, fs=sfreq, nperseg=nperseg)
def bandpower(freqs, psd, band):
idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
return trapz(psd[idx_band], freqs[idx_band])
def calculate_features_sliding_window(filtered_eeg_data, time_range):
"""
スライディングウィンドウ法で全バンドの特徴量を計算する
"""
from preprocess import create_epochs
all_features = []
eeg_stream = filtered_eeg_data['eeg_stream']
sfreq = int(eeg_stream['sfreq'])

window_size_sec = 0.5
step_size_sec = 0.1
window_samples = int(window_size_sec * sfreq)
step_samples = int(step_size_sec * sfreq)

# ★★ ここを修正 ★★
# 全てのバンドを定義
bands = {
    'delta': [1, 4],
    'theta': [4, 7],
    'alpha': [8, 13],
    'beta':  [13, 30],
    'gamma': [30, 50]
}
# ★★ ここまで ★★

for marker_val in filtered_eeg_data['markers']['marker_value'].unique():
    epoch = create_epochs(filtered_eeg_data, marker_val, time_range)
    if epoch is None or epoch['data'].shape[1] < window_samples: continue
    
    data = epoch['data']
    num_windows = (data.shape[1] - window_samples) // step_samples + 1

    for i in range(num_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        window_data = data[:, start_idx:end_idx]
        
        window_start_sec = start_idx / sfreq

        features = {'img_id': marker_val, 'window_start_sec': window_start_sec, 'window_end_sec': end_idx / sfreq}
        
        for ch_idx, ch_name in enumerate(['Fp1', 'Fp2']):
            ch_data = window_data[ch_idx]
            features[f'{ch_name}_amplitude'] = np.ptp(ch_data)
            
            freqs, psd = calculate_psd(ch_data, sfreq)
            # ★★ ここを修正 ★★
            # 全てのバンドパワーを計算
            for band_name, band_range in bands.items():
                features[f'{ch_name}_{band_name}'] = bandpower(freqs, psd, band_range)
            # ★★ ここまで ★★
        
        all_features.append(features)
        
if not all_features: return pd.DataFrame()
return pd.DataFrame(all_features)
