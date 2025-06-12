import numpy as np
import pandas as pd
from scipy import signal
from numpy import trapz
import streamlit as st

def calculate_psd(data, sfreq):
    """Welch法でPSDを計算"""
    nperseg = min(int(sfreq), data.shape[1]) # 1秒 or データ長
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    return freqs, psd

def bandpower(freqs, psd, band):
    """指定帯域のパワーを計算"""
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    # 積分してパワーを計算
    return trapz(psd[:, idx_band], freqs[idx_band], axis=1)

def calculate_features(filtered_eeg_data, eval_data, time_range):
    """
    1秒ごとのEEG特徴量を計算する
    """
    from preprocess import create_epochs # 循環参照を回避

    all_features = []
    
    # 解析対象の画像IDを取得
    img_ids = eval_data['img_id'].unique()
    
    # 周波数帯域の定義
    bands = {
        'delta': [1, 4],
        'theta': [4, 7],
        'alpha': [8, 13], # 8-13Hzに変更
        'beta':  [13, 30],
        'gamma': [30, 50]  # gammaを追加
    }

    # 各画像（トライアル）に対して処理
    for img_id in img_ids:
        epoch = create_epochs(filtered_eeg_data, img_id, time_range)
        if epoch is None: continue

        data = epoch['data']
        sfreq = int(epoch['sfreq'])
        
        # 1秒ごとに区切って特徴量を計算
        num_seconds = data.shape[1] // sfreq
        for sec in range(num_seconds):
            start_idx = sec * sfreq
            end_idx = (sec + 1) * sfreq
            
            # 1秒分のデータスライス
            second_data = data[:, start_idx:end_idx]
            
            # PSD計算
            freqs, psd = calculate_psd(second_data, sfreq)
            
            # 各バンドのパワーを計算
            features = {'img_id': img_id, 'second': sec}
            for band_name, band_range in bands.items():
                power = bandpower(freqs, psd, band_range)
                features[f'Fp1_{band_name}'] = power[0]
                features[f'Fp2_{band_name}'] = power[1]
            
            all_features.append(features)

    if not all_features:
        st.warning("特徴量を計算できるデータがありませんでした。")
        return pd.DataFrame()
        
    return pd.DataFrame(all_features)
