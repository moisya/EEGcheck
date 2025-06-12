import numpy as np
import pandas as pd
from scipy import signal
from numpy import trapz
import streamlit as st

def calculate_psd(data, sfreq):
    nperseg = min(len(data), sfreq)
    return signal.welch(data, fs=sfreq, nperseg=nperseg)

def bandpower(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return trapz(psd[idx_band], freqs[idx_band])

def calculate_features_sliding_window(filtered_eeg_data, time_range):
    """
    スライディングウィンドウ法で、より精密な特徴量を計算する
    """
    from preprocess import create_epochs

    all_features = []
    eeg_stream = filtered_eeg_data['eeg_stream']
    sfreq = int(eeg_stream['sfreq'])

    # スライディングウィンドウの設定
    window_size_sec = 0.25  # 窓の幅（秒）
    step_size_sec = 0.1   # 窓をスライドさせる幅（秒）
    window_samples = int(window_size_sec * sfreq)
    step_samples = int(step_size_sec * sfreq)

    bands = {
        'delta': [1, 4],
        'gamma': [30, 50]
    }

    # 全てのマーカー（トライアル）に対して処理
    for marker_val in filtered_eeg_data['markers']['marker_value'].unique():
        epoch = create_epochs(filtered_eeg_data, marker_val, time_range)
        if epoch is None: continue

        data = epoch['data']
        num_windows = (data.shape[1] - window_samples) // step_samples + 1

        for i in range(num_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            window_data = data[:, start_idx:end_idx]
            
            # このウィンドウの開始時間（秒）
            window_start_sec = start_idx / sfreq

            features = {
                'img_id': marker_val,
                'window_start_sec': window_start_sec,
                'window_end_sec': end_idx / sfreq
            }
            
            # 各チャンネルに対して特徴量を計算
            for ch_idx, ch_name in enumerate(['Fp1', 'Fp2']):
                ch_data = window_data[ch_idx]
                
                # 1. 振幅 (Peak-to-Peak)
                features[f'{ch_name}_amplitude'] = np.ptp(ch_data)
                
                # 2. バンドパワー
                freqs, psd = calculate_psd(ch_data, sfreq)
                features[f'{ch_name}_delta'] = bandpower(freqs, psd, bands['delta'])
                features[f'{ch_name}_gamma'] = bandpower(freqs, psd, bands['gamma'])
            
            all_features.append(features)
            
    if not all_features:
        st.warning("特徴量を計算できるデータがありませんでした。")
        return pd.DataFrame()
        
    return pd.DataFrame(all_features)
