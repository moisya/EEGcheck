import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
import streamlit as st
import copy

def apply_filters(eeg_data, freq_range, apply_notch=True):
    """EEGデータにフィルターを適用"""
    filtered_data = copy.deepcopy(eeg_data)
    signal_data = filtered_data['eeg_stream']['data']
    sfreq = filtered_data['eeg_stream']['sfreq']
    nyquist = 0.5 * sfreq

    low, high = freq_range
    if high >= nyquist:
        st.error(f"高域カットオフ周波数がナイキスト周波数({nyquist}Hz)以上です。")
        high = nyquist - 0.1
    if low <= 0:
        low = 0.1
    
    # 4次Butterworthバンドパスフィルター
    sos = butter(4, [low, high], btype='band', fs=sfreq, output='sos')
    filtered_signal = sosfiltfilt(sos, signal_data, axis=1)

    # 50Hzノッチフィルター
    if apply_notch:
        b_notch, a_notch = iirnotch(50.0, Q=30, fs=sfreq)
        filtered_signal = filtfilt(b_notch, a_notch, filtered_signal, axis=1)
    
    filtered_data['eeg_stream']['data'] = filtered_signal
    return filtered_data

def create_epochs(eeg_data, target_img_id, time_range):
    """特定の画像IDに対するエポックを作成"""
    markers = eeg_data['markers']
    if markers.empty or target_img_id not in markers['marker_value'].values:
        return None

    marker_time = markers[markers['marker_value'] == target_img_id].iloc[0]['marker_time']
    
    signal_data = eeg_data['eeg_stream']['data']
    time_stamps = eeg_data['eeg_stream']['times']
    sfreq = eeg_data['eeg_stream']['sfreq']

    start_time = marker_time + time_range[0]
    end_time = marker_time + time_range[1]

    start_idx = np.searchsorted(time_stamps, start_time, side='left')
    end_idx = np.searchsorted(time_stamps, end_time, side='right')

    if start_idx >= end_idx:
        return None

    epoch_data = signal_data[:, start_idx:end_idx]
    epoch_times = time_stamps[start_idx:end_idx] - marker_time
    
    return {
        'data': epoch_data,
        'times': epoch_times,
        'sfreq': sfreq,
        'img_id': target_img_id
    }
