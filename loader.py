import pandas as pd
import numpy as np
import pyxdf
import streamlit as st
import json
import tempfile
import os

@st.cache_data(show_spinner="XDFファイルを解析中...")
def load_xdf(uploaded_file):
    """
    特定の形式のXDFファイル（ラベル無し、先頭2chがEEG）を読み込むことに特化したローダー。
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xdf') as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    streams, _ = pyxdf.load_xdf(path)
    os.unlink(path)

    eeg_stream, marker_stream = None, None
    
    for s in streams:
        stream_type = s['info']['type'][0].lower()
        
        # EEGストリームを処理
        if 'eeg' in stream_type:
            st.info("先頭2チャンネルをEEGデータ(Fp1, Fp2)として読み込みます。")
            
            # チャンネル数が2つ以上あるか安全に確認
            if s['time_series'].shape[1] >= 2:
                eeg_indices = [0, 1] # 先頭2チャンネルに決め打ち
                eeg_stream = {
                    'data': s['time_series'][:, eeg_indices].T,
                    'times': s['time_stamps'],
                    'sfreq': float(s['info']['nominal_srate'][0]),
                    'ch_names': ['Fp1', 'Fp2'] # 名前はFp1, Fp2に固定
                }
            else:
                st.error("EEGストリームに2チャンネル分のデータがありません。")
                return None
        
        # マーカーを処理
        elif stream_type in ['markers', 'marker']:
            rows = []
            for ts, val_list in zip(s['time_stamps'], s['time_series']):
                if not val_list or not val_list[0]: continue
                val = val_list[0]
                try:
                    # JSON形式のマーカーを試す
                    obj = json.loads(val)
                    img_id = obj.get('img_id')
                    if img_id is not None:
                        rows.append({'marker_time': ts, 'marker_value': int(img_id)})
                except:
                    # JSONでなければ、単純な整数マーカーを試す
                    try:
                        rows.append({'marker_time': ts, 'marker_value': int(val)})
                    except:
                        continue # どちらでもなければ無視
            
            if rows:
                marker_stream = pd.DataFrame(rows)

    # 最終チェック
    if eeg_stream is None:
        st.error("EEGストリームが見つかりませんでした。")
        return None
    if marker_stream is None:
        st.warning("マーカーストリームが見つかりませんでした。")
        marker_stream = pd.DataFrame(columns=['marker_time', 'marker_value'])

    st.success(f"EEGデータ読み込み完了 (SampleRate: {eeg_stream['sfreq']} Hz)")
    return {'eeg_stream': eeg_stream, 'markers': marker_stream}


@st.cache_data(show_spinner="評価データを解析中...")
def load_evaluation_data(uploaded_file):
    """評価データ(CSV/XLSX)を読み込む（この部分は変更なし）"""
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("サポートされていないファイル形式です。")
            return None
        df.columns = df.columns.str.strip()
        if 'img_id' not in df.columns:
            st.error("評価データに必須列 'img_id' が見つかりません。")
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
