import streamlit as st
import os
import pandas as pd
from loader import load_xdf, load_evaluation_data
from preprocess import apply_filters, create_epochs
from features import calculate_features
from utils_plot import plot_waveforms, plot_outlier_scatter

# --- 初期設定と認証 ---
st.set_page_config(page_title="EEG Analysis App", page_icon="🧠", layout="wide")
def check_password():
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if st.session_state.authenticated: return True
    st.title("🧠 EEG Analysis App"); st.markdown("---")
    try: expected_password = st.secrets["APP_PASSWORD"]
    except: expected_password = os.getenv("APP_PASSWORD", "eeg2024")
    password = st.text_input("パスワードを入力", type="password")
    if st.button("ログイン"):
        if password == expected_password:
            st.session_state.authenticated = True; st.rerun()
        else: st.error("パスワードが違います")
    st.stop()

# --- セッション状態管理 ---
def initialize_session_state():
    keys = ["eeg_data", "eval_data", "raw_features_df", "filtered_features_df", "outlier_info_df"]
    for key in keys:
        if key not in st.session_state: st.session_state[key] = None

# --- サイドバーUI ---
def sidebar_controls():
    st.sidebar.title("📁 ファイル")
    xdf_file = st.sidebar.file_uploader("1. XDFファイル", type=['xdf'])
    eval_file = st.sidebar.file_uploader("2. 試行情報ファイル", type=['csv', 'xlsx'])
    
    if xdf_file and st.session_state.eeg_data is None:
        st.session_state.eeg_data = load_xdf(xdf_file)
    if eval_file and st.session_state.eval_data is None:
        st.session_state.eval_data = load_evaluation_data(eval_file)
    
    st.sidebar.markdown("---")
    st.sidebar.title("🔧 フィルター設定")
    freq_range = st.sidebar.slider("バンドパス (Hz)", 0.5, 60.0, (1.0, 50.0), 0.5)
    notch_filter = st.sidebar.checkbox("50Hz ノッチ", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.title("⏰ 解析時間範囲")
    time_range = st.sidebar.slider("マーカーからの時間(秒)", -5.0, 15.0, (0.0, 10.0), 0.5, help="特徴量計算と波形表示の基本範囲です")

    return {'freq_range': freq_range, 'notch_filter': notch_filter, 'time_range': time_range}

# --- タブ1: 波形ビューア ---
def waveform_viewer_tab(controls):
    st.header("① フィルター効果の確認")
    if st.session_state.eeg_data is None:
        st.warning("XDFファイルをアップロードしてください。"); return

    filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
    available_ids = st.session_state.eeg_data['markers']['marker_value'].unique()
    img_id = st.selectbox("表示する画像IDを選択", available_ids, key="tab1_img_id")
    
    raw_epoch = create_epochs(st.session_state.eeg_data, img_id, controls['time_range'])
    filtered_epoch = create_epochs(filtered_eeg, img_id, controls['time_range'])

    if raw_epoch and filtered_epoch:
        display_mode = st.radio("表示形式", ["重ねて", "並べて"], horizontal=True)
        plot_data = {'raw': raw_epoch['data'], 'filtered': filtered_epoch['data'], 'times': raw_epoch['times'], 'time_range': controls['time_range']}
        fig = plot_waveforms(plot_data, display_mode)
        st.plotly_chart(fig, use_container_width=True)

# --- タブ2: 外れ値除去 ---
def outlier_rejection_tab(controls):
    st.header("② 外れ値の検出と除去")
    if st.session_state.eeg_data is None or st.session_state.eval_data is None:
        st.warning("XDFファイルと試行情報ファイルの両方が必要です。"); return
    
    if st.button("📈 1秒ごとの特徴量を計算", type="primary"):
        with st.spinner("特徴量を計算中..."):
            filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
            features_df = calculate_features(filtered_eeg, st.session_state.eval_data, controls['time_range'])
            st.session_state.raw_features_df = features_df
            st.session_state.filtered_features_df = features_df # 初期状態は全データ
            st.session_state.outlier_info_df = pd.DataFrame() # 初期化
            st.success(f"{len(features_df)}個のデータポイント（1秒毎）が生成されました。")

    if st.session_state.raw_features_df is None:
        st.info("上のボタンを押して、特徴量計算を開始してください。"); return

    st.markdown("---"); st.subheader("📊 散布図による外れ値の可視化")
    df = st.session_state.raw_features_df
    feature_cols = [col for col in df.columns if col not in ['img_id', 'second']]
    eval_cols = st.session_state.eval_data.columns.tolist()

    col1, col2, col3 = st.columns(3)
    x_axis = col1.selectbox("X軸", feature_cols, 0)
    y_axis = col2.selectbox("Y軸", feature_cols, 1)
    color_axis = col3.selectbox("凡例/色", eval_cols, index=eval_cols.index('Dislike_Like') if 'Dislike_Like' in eval_cols else 0)

    # 主観評価データを特徴量dfにマージ
    plot_df = pd.merge(df, st.session_state.eval_data, on='img_id', how='left')

    st.markdown("##### 除去する閾値を設定")
    col1, col2, _, col3 = st.columns([2, 2, 1, 1])
    x_thresh = col1.number_input(f"X軸 ({x_axis}) の上限値", value=df[x_axis].quantile(0.999))
    y_thresh = col2.number_input(f"Y軸 ({y_axis}) の上限値", value=df[y_axis].quantile(0.999))
    if col3.button("閾値を適用"):
        outliers = df[(df[x_axis] >= x_thresh) | (df[y_axis] >= y_thresh)]
        st.session_state.outlier_info_df = outliers[['img_id', 'second']]
        st.session_state.filtered_features_df = df.drop(outliers.index)
    
    original_count = len(df)
    removed_count = len(st.session_state.outlier_info_df)
    filtered_count = original_count - removed_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("元の点数", original_count)
    col2.metric("除去された点数", removed_count, delta=-removed_count)
    col3.metric("残りの点数", filtered_count)
    
    fig = plot_outlier_scatter(plot_df, x_axis, y_axis, color_axis, x_thresh, y_thresh)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 除去後のデータテーブル")
    st.dataframe(st.session_state.filtered_features_df)
    st.download_button("📥 除去後データをCSVでダウンロード", st.session_state.filtered_features_df.to_csv(index=False).encode('utf-8-sig'), "filtered_features.csv", "text/csv")


# --- タブ3: 除去後波形 ---
def post_rejection_viewer_tab(controls):
    st.header("③ 除去後の波形確認")
    if st.session_state.outlier_info_df is None or st.session_state.outlier_info_df.empty:
        st.info("タブ②で外れ値を除去すると、ここに結果が表示されます。"); return

    filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
    outlier_img_ids = st.session_state.outlier_info_df['img_id'].unique()
    img_id_to_view = st.selectbox("確認する画像IDを選択", outlier_img_ids, key="tab3_img_id")
    
    st.info(f"画像ID: {img_id_to_view} の波形。赤色でハイライトされた区間が、タブ②で外れ値として除去された1秒間です。")
    
    raw_epoch = create_epochs(st.session_state.eeg_data, img_id_to_view, controls['time_range'])
    filtered_epoch = create_epochs(filtered_eeg, img_id_to_view, controls['time_range'])
    
    if raw_epoch and filtered_epoch:
        plot_data = {'raw': raw_epoch['data'], 'filtered': filtered_epoch['data'], 'times': raw_epoch['times'], 'time_range': controls['time_range']}
        # この画像IDに関する外れ値情報だけを渡す
        outliers_for_plot = st.session_state.outlier_info_df[st.session_state.outlier_info_df['img_id'] == img_id_to_view]
        fig = plot_waveforms(plot_data, display_mode="並べて", outlier_df=outliers_for_plot)
        st.plotly_chart(fig, use_container_width=True)


# --- メイン実行部 ---
def main():
    check_password()
    initialize_session_state()
    st.title("🧠 EEG Analysis App for Outlier Rejection")
    controls = sidebar_controls()
    
    tab1, tab2, tab3 = st.tabs(["① フィルター効果の確認", "② 外れ値の検出・除去", "③ 除去後の波形確認"])
    with tab1:
        waveform_viewer_tab(controls)
    with tab2:
        outlier_rejection_tab(controls)
    with tab3:
        post_rejection_viewer_tab(controls)

if __name__ == "__main__":
    main()
