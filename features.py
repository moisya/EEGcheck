import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapz
import streamlit as st

def calculate_psd(data, sfreq):
    """Welch法でPSDを計算"""
    nperseg = min(int(sfreq * 2), data.shape[1]) # 2秒 or データ長
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    return freqs, psd

def bandpower(freqs, psd, band):
    """指定帯域のパワーを計算"""
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return trapz(psd[:, idx_band], freqs[idx_band], axis=1)

def calculate_features(filtered_eeg_data, eval_data, time_range):
    """全トライアルの特徴量を計算し、評価データと結合"""
    from preprocess import create_epochs # 関数内インポートで循環参照を回避

    all_features = []
    
    img_ids = eval_data['img_id'].unique()
    progress_bar = st.progress(0)
    
    for i, img_id in enumerate(img_ids):
        epoch = create_epochs(filtered_eeg_data, img_id, time_range)
        if epoch is None or epoch['data'].shape[1] < int(epoch['sfreq']): # 1秒未満のデータは除外
            continue

        freqs, psd = calculate_psd(epoch['data'], epoch['sfreq'])
        
        # パワー計算
        theta_power = bandpower(freqs, psd, [4, 7])
        alpha_power = bandpower(freqs, psd, [8, 12])
        beta_power = bandpower(freqs, psd, [13, 30])
        total_power = bandpower(freqs, psd, [1, 40])
        
        # ゼロ除算を回避
        total_power[total_power == 0] = 1e-10

        # ピークアルファ周波数
        alpha_mask = np.logical_and(freqs >= 6, freqs <= 14)
        peak_alpha_freq = [freqs[alpha_mask][np.argmax(p[alpha_mask])] if np.any(alpha_mask) else np.nan for p in psd]

        features = {
            'img_id': img_id,
            'alpha_power_Fp1': alpha_power[0],
            'alpha_power_Fp2': alpha_power[1],
            'beta_power_Fp1': beta_power[0],
            'beta_power_Fp2': beta_power[1],
            'theta_power_Fp1': theta_power[0],
            'theta_power_Fp2': theta_power[1],
            'alpha_rel_power_Fp1': alpha_power[0] / total_power[0],
            'alpha_rel_power_Fp2': alpha_power[1] / total_power[1],
            'beta_rel_power_Fp1': beta_power[0] / total_power[0],
            'beta_rel_power_Fp2': beta_power[1] / total_power[1],
            'theta_rel_power_Fp1': theta_power[0] / total_power[0],
            'theta_rel_power_Fp2': theta_power[1] / total_power[1],
            'alpha_asymmetry': np.log10(alpha_power[1]) - np.log10(alpha_power[0]),
            'beta_asymmetry': np.log10(beta_power[1]) - np.log10(beta_power[0]),
            'theta_asymmetry': np.log10(theta_power[1]) - np.log10(theta_power[0]),
            'peak_alpha_freq_Fp1': peak_alpha_freq[0],
            'peak_alpha_freq_Fp2': peak_alpha_freq[1],
        }
        all_features.append(features)
        progress_bar.progress((i + 1) / len(img_ids))

    if not all_features:
        return None
        
    features_df = pd.DataFrame(all_features)
    return pd.merge(features_df, eval_data, on='img_id', how='inner')
