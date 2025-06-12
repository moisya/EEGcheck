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
        st.warning("Streamlit CloudのSecretsに'APP_PASSWORD'が設定されていません。開発用のデフォルトパスワード 'eeg2024' を使用します。")
        expected_password = os.getenv("APP_PASSWORD", "eeg2024")

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
