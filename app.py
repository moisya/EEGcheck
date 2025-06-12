ご依頼ありがとうございます。2チャンネルEEGデータの解析・可視化Streamlitアプリを、ご提示の要件と途中経過のコードを基に完成させました。

各ファイルはモジュール構成の推奨に従い、インタラクションの最適化やデプロイを考慮して実装しています。

成果物

以下に、GitHubリポジトリ一式に含めるべき全ファイル（ソースコード、README、requirements.txt）を記載します。

1. ソースコード (app.py, loader.py, preprocess.py, features.py, utils_plot.py)

提供されたコードを基に、データフローの修正、UIの完成、インタラクションの最適化を行いました。

import streamlit as st
import os
import pandas as pd
import numpy as np
from loader import load_xdf_data, load_evaluation_data
from preprocess import apply_filters, create_epochs
from features import calculate_features
from utils_plot import plot_waveforms, plot_scatter_with_regression
import plotly.graph_objects as go

# ページ設定
st.set_page_config(
    page_title="EEG Analysis App",
    page_icon="🧠",
    layout="wide"
)

# --- 認証機能 ---
def check_password():
    """簡易パスワード認証"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🧠 EEG Analysis App")
    st.markdown("---")

    try:
        expected_password = st.secrets["APP_PASSWORD"]
    except:
        st.warning("Streamlit CloudのSecretsに'APP_PASSWORD'が設定されていません。開発用のデフォルトパスワード 'eeg2025' を使用します。")
        expected_password = os.getenv("APP_PASSWORD", "eeg2025")

    password = st.text_input("パスワードを入力してください", type="password")

    if st.button("ログイン"):
        if password == expected_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが正しくありません。")
            
    st.stop()


# --- セッション状態管理 ---
def initialize_session_state():
    """セッション状態の初期化"""
    if "eeg_data" not in st.session_state:
        st.session_state.eeg_data = None
    if "eval_data" not in st.session_state:
        st.session_state.eval_data = None
    if "feature_df" not in st.session_state:
        st.session_state.feature_df = None
    if "last_filter_settings" not in st.session_state:
        st.session_state.last_filter_settings = {}

# --- サイドバーUI ---
def sidebar_controls():
    """サイドバーのコントロールウィジェットを配置"""
    st.sidebar.title("📁 ファイルアップロード")
    xdf_file = st.sidebar.file_uploader("1. XDFファイルをアップロード", type=['xdf'])
    eval_file = st.sidebar.file_uploader("2. 評価データ (CSV/XLSX) をアップロード", type=['csv', 'xlsx'])

    if xdf_file and st.session_state.eeg_data is None:
        with st.spinner("XDFファイルを読み込み中..."):
            st.session_state.eeg_data = load_xdf_data(xdf_file)
    if eval_file and st.session_state.eval_data is None:
        with st.spinner("評価データを読み込み中..."):
            st.session_state.eval_data = load_evaluation_data(eval_file)

    st.sidebar.markdown("---")
    st.sidebar.title("🔧 フィルター設定")
    freq_range = st.sidebar.slider("バンドパスフィルター (Hz)", 0.5, 60.0, (1.0, 40.0), 0.5)
    notch_filter = st.sidebar.checkbox("50Hz ノッチフィルターを適用", value=True)

    st.sidebar.markdown("---")
    st.sidebar.title("📊 表示・解析範囲設定")
    range_type = st.sidebar.radio("範囲指定方式", ["画像ID", "秒数（データ先頭から）"])

    time_range, img_id = None, None
    if range_type == "画像ID":
        if st.session_state.eeg_data and st.session_state.eeg_data['markers'] is not None and not st.session_state.eeg_data['markers'].empty:
            available_ids = sorted(st.session_state.eeg_data['markers']['marker_value'].unique())
            img_id = st.sidebar.selectbox("画像IDを選択", available_ids)
        else:
            st.sidebar.warning("マーカーが見つからないため、画像IDを選択できません。")
        time_range = st.sidebar.slider("マーカー前後の時間 (秒)", -5.0, 10.0, (-1.0, 4.0), 0.1)

    else: # 秒数指定
        max_duration = 0
        if st.session_state.eeg_data:
            times = st.session_state.eeg_data['eeg_stream']['times']
            max_duration = times[-1] - times[0]
        
        start_val, end_val = st.sidebar.slider(
            "表示時間範囲 (秒)", 0.0, float(max_duration), (0.0, min(10.0, max_duration)), 0.5
        )
        time_range = (start_val, end_val)


    # フィルター設定が変更されたら、計算済み特徴量をリセット
    current_settings = {"freq": freq_range, "notch": notch_filter}
    if st.session_state.last_filter_settings != current_settings:
        st.session_state.feature_df = None
        st.session_state.last_filter_settings = current_settings
        if "rerun_warning_shown" not in st.session_state:
             st.toast("フィルター設定が変更されました。特徴量は再計算が必要です。")
             st.session_state.rerun_warning_shown = True


    return {
        'freq_range': freq_range, 'notch_filter': notch_filter,
        'time_range': time_range, 'img_id': img_id, 'range_type': range_type
    }

# --- タブ別コンテンツ ---
def waveform_viewer_tab(controls):
    st.header("📈 波形ビューア")
    if not st.session_state.eeg_data:
        st.warning("サイドバーからXDFファイルをアップロードしてください。")
        return

    display_mode = st.radio("表示形式", ["重ねて", "並べて"], key="display_mode")

    with st.spinner("フィルター適用中..."):
        filtered_eeg_data = apply_filters(
            st.session_state.eeg_data, controls['freq_range'], controls['notch_filter']
        )

    plot_data = None
    if controls['range_type'] == "画像ID" and controls['img_id'] is not None:
        with st.spinner(f"画像ID {controls['img_id']} のエポックを作成中..."):
            raw_epoch = create_epochs(st.session_state.eeg_data, controls['img_id'], controls['time_range'])
            filtered_epoch = create_epochs(filtered_eeg_data, controls['img_id'], controls['time_range'])
        
        if raw_epoch and filtered_epoch:
            plot_data = {'raw': raw_epoch['data'], 'filtered': filtered_epoch['data'], 'times': raw_epoch['times']}
        else:
            st.error("エポック作成に失敗しました。")
            return

    elif controls['range_type'] == "秒数（データ先頭から）":
        raw_stream = st.session_state.eeg_data['eeg_stream']
        filtered_stream = filtered_eeg_data['eeg_stream']
        times, sfreq = raw_stream['times'], raw_stream['sfreq']
        
        start_sec, end_sec = controls['time_range']
        start_idx = int(start_sec * sfreq)
        end_idx = int(end_sec * sfreq)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(times), end_idx)

        if start_idx < end_idx:
            sliced_times = (np.arange(end_idx - start_idx) / sfreq) + start_sec
            plot_data = {
                'raw': raw_stream['data'][:, start_idx:end_idx],
                'filtered': filtered_stream['data'][:, start_idx:end_idx],
                'times': sliced_times
            }
        else:
            st.warning("指定された時間範囲にデータがありません。")

    if plot_data:
        fig = plot_waveforms(plot_data, display_mode)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("サイドバーで表示範囲を指定してください。")


def frequency_analysis_tab(controls):
    st.header("🔬 周波数解析・散布図")
    if not st.session_state.eeg_data or not st.session_state.eval_data:
        st.warning("XDFファイルと評価データの両方をアップロードしてください。")
        return

    if st.button("🚀 周波数解析を実行", type="primary"):
        with st.spinner("特徴量を計算中...（トライアル数により時間がかかります）"):
            filtered_eeg_data = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
            st.session_state.feature_df = calculate_features(filtered_eeg_data, st.session_state.eval_data, controls['time_range'])
        if st.session_state.feature_df is not None:
            st.success("特徴量の計算が完了しました！")
        else:
            st.error("特徴量の計算に失敗しました。")

    if st.session_state.feature_df is not None:
        df = st.session_state.feature_df
        st.markdown("---")
        st.subheader("📊 散布図と相関分析")
        
        col1, col2 = st.columns(2)
        feature_cols = sorted([c for c in df.columns if c not in ['sid', 'img_id', 'time', 'Dislike_Like', 'sam_val', 'sam_aro']])
        eval_cols = sorted([c for c in ['Dislike_Like', 'sam_val', 'sam_aro'] if c in df.columns])

        with col1:
            x_axis = st.selectbox("X軸（EEG特徴量）", feature_cols, index=feature_cols.index("alpha_asymmetry"))
        with col2:
            if not eval_cols:
                st.error("評価データに 'Dislike_Like', 'sam_val', 'sam_aro' のいずれかの列が見つかりません。")
                return
            y_axis = st.selectbox("Y軸（主観評価）", eval_cols)

        if x_axis and y_axis:
            fig, r, p = plot_scatter_with_regression(df, x_axis, y_axis)
            if fig:
                col1, col2 = st.columns(2)
                col1.metric("ピアソンの相関係数 (r)", f"{r:.3f}")
                significance = "p < 0.05 (有意)" if p < 0.05 else f"p = {p:.3f} (非有意)"
                col2.metric("p値", significance)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 特徴量・評定データテーブル")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 テーブルをCSVでダウンロード", csv, f"eeg_features_{xdf_file.name if 'xdf_file' in locals() else ''}.csv", "text/csv")


# --- メイン実行部 ---
def main():
    if not check_password():
        return

    initialize_session_state()
    
    st.title("🧠 EEG Analysis App")
    st.markdown("2チャンネルEEG（Fp1, Fp2）の波形比較と周波数解析を行います。")

    controls = sidebar_controls()

    tab1, tab2 = st.tabs(["📈 波形ビューア", "🔬 周波数解析・散布図"])
    with tab1:
        waveform_viewer_tab(controls)
    with tab2:
        frequency_analysis_tab(controls)

if __name__ == "__main__":
    main()


提供されたコードを基に、サンプリングレートのチェック機能を追加しました。

import pandas as pd
import numpy as np
import pyxdf
import streamlit as st
from io import BytesIO
import tempfile
import os

@st.cache_data(show_spinner="XDFファイルを解析中...")
def load_xdf_data(uploaded_file):
    """XDFファイルからEEGデータとマーカーを読み込む"""
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
                markers = [{'marker_time': ts, 'marker_value': int(val[0])}
                           for ts, val in zip(stream['time_stamps'], stream['time_series'])]
                marker_stream = pd.DataFrame(markers)

        if eeg_stream is None:
            st.error("XDFファイルにFp1, Fp2チャンネルを含むEEGストリームが見つかりません。")
            return None
        if marker_stream is None:
            st.warning("マーカーストリームが見つかりません。波形ビューア（秒数指定）のみ利用可能です。")
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
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

提供されたコードから、依頼内容に直接関係のない関数を削除し、見通しを良くしました。

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
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

提供されたコードを基に、循環参照を避けるためのインポート方法を調整し、依頼仕様に合わせた特徴量計算ロジックを確定させました。

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
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

提供されたコードを完成させ、散布図描画機能を追加しました。

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
import streamlit as st

def plot_waveforms(plot_data, display_mode="重ねて"):
    """生波形とフィルター後波形をプロット"""
    raw, filtered, times = plot_data['raw'], plot_data['filtered'], plot_data['times']
    ch_names = ['Fp1', 'Fp2']
    colors = px.colors.qualitative.Plotly

    if display_mode == "重ねて":
        fig = go.Figure()
        for i, ch in enumerate(ch_names):
            fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', line_color=colors[i], opacity=0.4))
            fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch} (フィルター後)', line_color=colors[i]))
        fig.update_layout(title="EEG波形比較（重ねて表示）", xaxis_title="時間 (秒)", yaxis_title="振幅 (μV)", legend_orientation="h")
    
    else: # 並べて
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=ch_names, vertical_spacing=0.1)
        for i, ch in enumerate(ch_names):
            fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', legendgroup=ch, line_color=colors[i], opacity=0.4, showlegend=(i==0)), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch} (フィルター後)', legendgroup=ch, line_color=colors[i], showlegend=(i==0)), row=i+1, col=1)
        fig.update_layout(title="EEG波形比較（並べて表示）", legend_orientation="h")
        fig.update_xaxes(title_text="時間 (秒)", row=2, col=1)
        fig.update_yaxes(title_text="振幅 (μV)")

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Marker")
    fig.update_layout(height=500, template="plotly_white", hovermode="x unified")
    return fig

def plot_scatter_with_regression(data, x_col, y_col):
    """散布図と回帰線をプロットし、相関係数とp値を返す"""
    clean_data = data[[x_col, y_col, 'img_id']].dropna()
    if len(clean_data) < 3:
        st.warning(f"有効なデータが3点未満のため、散布図を描画できません。")
        return None, np.nan, np.nan

    x, y = clean_data[x_col], clean_data[y_col]
    r, p = pearsonr(x, y)
    
    fig = px.scatter(clean_data, x=x_col, y=y_col,
                     trendline="ols", trendline_color_override="red",
                     hover_data=['img_id'],
                     title=f"<b>{y_col}</b> vs <b>{x_col}</b>")
    
    fig.update_layout(template="plotly_white", height=500)
    return fig, r, p
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
2. requirements.txt (必要ライブラリ)

このファイルをプロジェクトのルートに配置してください。

streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.20.0
scipy>=1.9.0
plotly>=5.10.0
pyxdf>=1.16.0
openpyxl>=3.0.0
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Txt
IGNORE_WHEN_COPYING_END
3. デプロイマニュアル (README.md)

このファイルをGitHubリポジトリのルートに配置してください。

# 🧠 2チャンネルEEG解析・可視化アプリ

2チャンネルEEG (Fp1, Fp2) のXDFデータと、同一トライアルの主観評価データを入力し、インタラクティブな波形比較と周波数解析を行うStreamlitアプリケーションです。

## ✨ 主な機能

- **ファイルアップロード**:
  - EEGデータ (`.xdf`形式)
  - 主観評価データ (`.csv` or `.xlsx`形式)
- **インタラクティブな波形ビューア**:
  - 生データとフィルター後データの比較（重ねて/並べて表示）
  - 表示範囲を画像IDまたは秒数で指定可能
- **周波数解析とEDA**:
  - Welch法によるパワースペクトル密度 (PSD) 特徴量の算出
    - 各バンドパワー（α, β, θ）と相対パワー
    - 前頭葉非対称性 (Frontal Asymmetry)
    - ピークアルファ周波数
  - EEG特徴量と主観評価の散布図表示（回帰直線付き）
  - ピアソン相関係数とp値の自動計算
  - 解析結果のテーブル表示とCSVダウンロード機能

## ⚙️ ローカルでの実行方法

1.  **リポジトリをクローン:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **必要なライブラリをインストール:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **環境変数を設定（任意）:**
    ローカルでパスワード認証をテストする場合、環境変数を設定します。
    ```bash
    # for Linux/macOS
    export APP_PASSWORD="your_local_password"
    
    # for Windows (Command Prompt)
    set APP_PASSWORD="your_local_password"
    ```
    設定しない場合、デフォルトのパスワード `eeg2025` が使用されます。

4.  **Streamlitアプリを実行:**
    ```bash
    streamlit run app.py
    ```
    ブラウザで `http://localhost:8501` が開きます。

## ☁️ Streamlit Community Cloudへのデプロイ

1.  **GitHubリポジトリの準備:**
    - このプロジェクトの全ファイル (`app.py`, `loader.py`, `preprocess.py`, `features.py`, `utils_plot.py`, `requirements.txt`, `README.md`) をGitHubリポジトリにプッシュします。

2.  **Streamlit Cloudにデプロイ:**
    - [Streamlit Community Cloud](https://share.streamlit.io/)にログインします。
    - "New app" ボタンをクリックし、"Deploy from GitHub" を選択します。
    - 作成したリポジトリ、ブランチ（例: `main`）、メインファイル（`app.py`）を選択します。

3.  **パスワード認証の設定 (Secrets):**
    - "Advanced settings..." をクリックします。
    - **Secrets** のテキストボックスに、アプリのパスワードをTOML形式で入力します。

      ```toml
      # .streamlit/secrets.toml
      APP_PASSWORD = "your_secure_password_here"
      ```

    - "Save" をクリックします。

4.  **デプロイ実行:**
    - "Deploy!" ボタンをクリックします。
    - デプロイが完了すると、あなたのアプリが公開されます。

## 📝 注意事項

- **XDFファイル**: `Fp1`, `Fp2` というラベルを持つチャンネルと、画像IDを値として持つマーカーストリームが必要です。サンプリング周波数は250Hz以上を推奨します。
- **評価データ**: `img_id` 列が必須です。この列をキーにしてEEGのマーカーと紐付けられます。
- **パフォーマンス**: 大量のトライアルを含むデータを解析する場合、特徴量計算に時間がかかることがあります。計算中はスピナーが表示されます。
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
