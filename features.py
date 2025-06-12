import numpy as np
import pandas as pd
from scipy import signal
from numpy import trapz
import streamlit as st

def calculate_psd(data, sfreq):
    # 窓の長さに合わせてnpersegを設定
    nperseg = len(data)
    return signal.welch(data, fs=sfreq, nperseg=nperseg)

def bandpower(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return trapz(psd[idx_band], freqs[idx_band])

def calculate_features_sliding_window(filtered_eeg_data, time_range):
    """
    スライディングウィンドウ法で、より精密な特徴量を計算する（最適化版）
    """
    from preprocess import create_epochs

    all_features = []
    eeg_stream = filtered_eeg_data['eeg_stream']
    sfreq = int(eeg_stream['sfreq'])

    # ★★ ここを修正 ★★
    # スライディングウィンドウの設定を最適化
    window_size_sec = 1.0  # 窓の幅を1秒に延長 -> 信頼できる周波数分解能を確保
    step_size_sec = 0.25   # スライド幅は短く保つ -> 時間的な精度を確保
    # ★★ ここまで ★★
    
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
        # エポック長がウィンドウサイズより短い場合はスキップ
        if data.shape[1] < window_samples: continue
        
        num_windows = (data.shape[1] - window_samples) // step_samples + 1

        for i in range(num_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            window_data = data[:, start_idx:end_idx]
            
            window_start_sec = start_idx / sfreq

            features = {
                'img_id': marker_val,
                'window_start_sec': window_start_sec,
                'window_end_sec': end_idx / sfreq
            }
            
            for ch_idx, ch_name in enumerate(['Fp1', 'Fp2']):
                ch_data = window_data[ch_idx]
                
                features[f'{ch_name}_amplitude'] = np.ptp(ch_data)
                
                freqs, psd = calculate_psd(ch_data, sfreq)
                features[f'{ch_name}_delta'] = bandpower(freqs, psd, bands['delta'])
                features[f'{ch_name}_gamma'] = bandpower(freqs, psd, bands['gamma'])
            
            all_features.append(features)
            
    if not all_features:
        st.warning("特徴量を計算できるデータがありませんでした。")
        return pd.DataFrame()
        
    return pd.DataFrame(all_features)
