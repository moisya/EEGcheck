import streamlit as st
import os
import pandas as pd
from loader import load_xdf, load_evaluation_data
from preprocess import apply_filters, create_epochs
from features import calculate_features_sliding_window
from utils_plot import plot_waveforms, plot_outlier_scatter

# --- 初期設定と認証 ---
st.set_page_config(page_title="EEG Precision Artifact Removal", page_icon="🧠", layout="wide")
def check_password():
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if st.session_state.authenticated: return True
    st.title("🧠 EEG Analysis App"); st.markdown("---")
    try: expected_password = st.secrets["APP_PASSWORD"]
    except: expected_password = os.getenv("APP_PASSWORD", "eeg2024")
    password = st.text_input("パスワードを入力", type="password")
    if st.button("ログイン"):
        if password == expected_password: st.session_state.authenticated = True; st.rerun()
        else: st.error("パスワードが違います")
    st.stop()

# --- セッション状態管理 ---
def initialize_session_state():
    keys = ["eeg_data", "eval_data", "features_df", "outlier_windows_df"]
    for key in keys:
        if key not in st.session_state: st.session_state[key] = None

# --- サイドバーUI ---
def sidebar_controls():
    st.sidebar.title("📁 ファイル")
    xdf_file = st.sidebar.file_uploader("1. XDFファイル", type=['xdf'])
    eval_file = st.sidebar.file_uploader("2. 試行情報ファイル", type=['csv', 'xlsx'])
    if xdf_file and st.session_state.eeg_data is None: st.session_state.eeg_data = load_xdf(xdf_file)
    if eval_file and st.session_state.eval_data is None: st.session_state.eval_data = load_evaluation_data(eval_file)
    
    st.sidebar.markdown("---"); st.sidebar.title("🔧 フィルター設定")
    freq_range = st.sidebar.slider("バンドパス (Hz)", 0.5, 60.0, (1.0, 50.0), 0.5)
    notch_filter = st.sidebar.checkbox("50Hz ノッチ", value=True)
    
    st.sidebar.markdown("---"); st.sidebar.title("⏰ 解析時間範囲")
    time_range = st.sidebar.slider("マーカーからの時間(秒)", -5.0, 15.0, (0.0, 10.0), 0.5, help="特徴量計算と波形表示の基本範囲です")
    return {'freq_range': freq_range, 'notch_filter': notch_filter, 'time_range': time_range}

# --- 外れ値除去タブ ---
def outlier_rejection_tab(controls):
    st.header("🔬 アーチファクトの検出と除去")
    if st.session_state.eeg_data is None: st.warning("XDFファイルをアップロードしてください。"); return
    
    if st.button("📈 精密スキャンを実行", type="primary"):
        with st.spinner("スライディングウィンドウで特徴量を計算中..."):
            filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
            features_df = calculate_features_sliding_window(filtered_eeg, controls['time_range'])
            st.session_state.features_df = features_df
            st.session_state.outlier_windows_df = pd.DataFrame()
            st.success(f"{len(features_df)}個の微小区間（ウィンドウ）が生成されました。")

    if st.session_state.features_df is None or st.session_state.features_df.empty:
        st.info("上のボタンを押して、特徴量計算を開始してください。"); return

    st.markdown("---"); st.subheader("📊 散布図によるアーチファクトの可視化と除去")
    df = st.session_state.features_df
    ch_select = st.radio("対象チャンネル", ["Fp1", "Fp2"], horizontal=True)
    
    st.markdown("##### 除去する閾値を設定（いずれか一つでも超えたら除去）")
    
    # 6つの閾値を設定
    thresholds = {}
    cols = st.columns(3)
    bands = ['amplitude', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    for i, band in enumerate(bands):
        key = f'{ch_select}_{band}'
        if key in df.columns:
            thresholds[key] = cols[i % 3].number_input(f"{key} の上限", value=df[key].quantile(0.99))

    # 外れ値ウィンドウを特定
    if thresholds:
      query = " | ".join([f"`{key}` >= {val}" for key, val in thresholds.items()])
      outliers = df.query(query)
      st.session_state.outlier_windows_df = outliers
      st.metric("除去された微小区間（ウィンドウ）の数", len(outliers), f"-{len(outliers) / len(df):.1%}" if len(df) > 0 else "")
    
    # 表示するグラフの軸を自由に選択
    st.markdown("##### 表示するグラフの軸を選択")
    feature_cols = [f'{ch_select}_{b}' for b in bands]
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("X軸", feature_cols, index=1) # delta
    y_axis = col2.selectbox("Y軸", feature_cols, index=0) # amplitude

    if x_axis in df.columns and y_axis in df.columns:
        fig = plot_outlier_scatter(df, x_axis, y_axis, thresholds.get(x_axis), thresholds.get(y_axis))
        st.plotly_chart(fig, use_container_width=True)

# --- 除去後波形タブ ---
def post_rejection_viewer_tab(controls):
    st.header("👀 除去後の波形確認")
    if st.session_state.outlier_windows_df is None or st.session_state.outlier_windows_df.empty:
        st.info("左のタブで閾値を設定すると、除去された区間がここに表示されます。"); return

    filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
    outlier_img_ids = st.session_state.outlier_windows_df['img_id'].unique()
    img_id_to_view = st.selectbox("確認する画像IDを選択", outlier_img_ids)
    
    st.info(f"画像ID: {img_id_to_view} の波形。赤色でハイライトされた区間が除去された微小区間です。")
    
    raw_epoch = create_epochs(st.session_state.eeg_data, img_id_to_view, controls['time_range'])
    filtered_epoch = create_epochs(filtered_eeg, img_id_to_view, controls['time_range'])
    
    if raw_epoch and filtered_epoch:
        plot_data = {'raw': raw_epoch['data'], 'filtered': filtered_epoch['data'], 'times': raw_epoch['times'], 'time_range': controls['time_range']}
        outliers_for_plot = st.session_state.outlier_windows_df[st.session_state.outlier_windows_df['img_id'] == img_id_to_view]
        
        outliers_for_plot_renamed = outliers_for_plot.rename(columns={'window_start_sec': 'second', 'window_end_sec': 'second_end'})
        
        fig = plot_waveforms(plot_data, display_mode="並べて", outlier_df=outliers_for_plot_renamed)
        st.plotly_chart(fig, use_container_width=True)

# --- メイン実行部 ---
def main():
    check_password()
    initialize_session_state()
    st.title("🧠 EEG 精密アーチファクト除去ツール")
    controls = sidebar_controls()
    
    tab1, tab2 = st.tabs(["🔬 アーチファクトの検出・除去", "👀 除去後の波形確認"])
    with tab1: outlier_rejection_tab(controls)
    with tab2: post_rejection_viewer_tab(controls)

if __name__ == "__main__": main()
