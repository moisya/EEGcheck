import pandas as pd
import numpy as np
import pyxdf
import streamlit as st
from io import BytesIO
import tempfile
import os

@st.cache_data(show_spinner="XDFファイルを解析中...")
def load_xdf_data(uploaded_file):
    """XDFファイルからEEGデータとマーカーを読み込む（堅牢版）"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        streams, _ = pyxdf.load_xdf(tmp_path)
        os.unlink(tmp_path)

        eeg_stream, marker_stream = None, None
        for stream in streams:
            info = stream['info']
            stype = info['type'][0].lower()
            
            # EEGストリームを特定 (Fp1, Fp2を含む)
            if 'channels' in info and 'channel' in info['channels'][0]:
                labels = [ch['label'][0] for ch in info['channels'][0]['channel']]
                if 'Fp1' in labels and 'Fp2' in labels:
                    if float(info['nominal_srate'][0]) < 250:
                        st.error(f"サンプリングレートが低すぎます: {info['nominal_srate'][0]} Hz. 250Hz以上が必要です。")
                        return None

                    fp1_idx, fp2_idx = labels.index('Fp1'), labels.index('Fp2')
                    eeg_stream = {
                        'data': stream['time_series'][:, [fp1_idx, fp2_idx]].T,
                        'times': stream['time_stamps'],
                        'sfreq': float(info['nominal_srate'][0]),
                        'ch_names': ['Fp1', 'Fp2']
                    }

            # マーカーストリームを特定
            elif stype in ['markers', 'marker']:
                marker_list = []
                for ts, val_list in zip(stream['time_stamps'], stream['time_series']):
                    if not val_list:  # 空のマーカーはスキップ
                        continue
                    try:
                        # 整数に変換できるマーカーのみを抽出
                        marker_value = int(val_list[0])
                        marker_list.append({'marker_time': ts, 'marker_value': marker_value})
                    except (ValueError, TypeError):
                        # 整数に変換できないマーカー（例：JSON文字列）は無視する
                        continue
                
                if marker_list:
                    marker_stream = pd.DataFrame(marker_list)

        if eeg_stream is None:
            st.error("XDFファイルにFp1, Fp2チャンネルを含むEEGストリームが見つかりません。")
            return None
        if marker_stream is None:
            st.warning("数字の形式のマーカーストリームが見つかりませんでした。波形ビューア（秒数指定）のみ利用可能です。")
            marker_stream = pd.DataFrame(columns=['marker_time', 'marker_value'])

        st.success(f"EEGデータ読み込み完了 (SampleRate: {eeg_stream['sfreq']} Hz)")
        return {'eeg_stream': eeg_stream, 'markers': marker_stream}

    except Exception as e:
        st.error(f"XDFファイルの読み込みに失敗しました: {e}")
        return None

@st.cache_data(show_spinner="評価データを解析中...")
def load_evaluation_data(uploaded_file):
    """評価データ(CSV/XLSX)を読み込む"""
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("サポートされていないファイル形式です。CSVまたはXLSXをアップロードしてください。")
            return None

        df.columns = df.columns.str.strip()
        required_cols = ['img_id']
        if not all(col in df.columns for col in required_cols):
            st.error(f"評価データに必須列 'img_id' が見つかりません。")
            return None

        df['img_id'] = pd.to_numeric(df['img_id'], errors='coerce').dropna().astype(int)
        
        for col in ['Dislike_Like', 'sam_val', 'sam_aro']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.success(f"評価データ読み込み完了 ({len(df)}件)")
        return df

    except Exception as e:
        st.error(f"評価データの読み込みに失敗しました: {e}")
        return None
