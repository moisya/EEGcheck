import streamlit as st
import os
import pandas as pd
from loader import load_xdf, load_evaluation_data
from preprocess import apply_filters
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
    keys = ["eeg_data", "eval_data", "raw_features_df", "filtered_features_df"]
    for key in keys:
        if key not in st.session_state: st.session_state[key] = None

# --- サイドバーUI ---
def sidebar_controls():
    st.sidebar.title("📁 ファイル")
    if st.sidebar.button("サンプルデータで試す"):
        # サンプルデータ作成ロジック（後で実装）
        pass
    xdf_file = st.sidebar.file_uploader("1. XDFファイル", type=['xdf'])
    if xdf_file: st.session_state.eeg_data = load_xdf(xdf_file)
    eval_file = st.sidebar.file_uploader("2. 試行情報ファイル", type=['csv', 'xlsx'])
    if eval_file: st.session_state.eval_data = load_evaluation_data(eval_file)
    
    st.sidebar.markdown("---")
    st.sidebar.title("🔧 フィルター設定")
    freq_range = st.sidebar.slider("バンドパス (Hz)", 0.5, 60.0, (1.0, 50.0), 0.5)
    notch_filter = st.sidebar.checkbox("50Hz ノッチ", value=True)
    return {'freq_range': freq_range, 'notch_filter': notch_filter}

# --- 外れ値除去タブ ---
def outlier_rejection_tab(controls):
    st.header("🔬 外れ値の検出と除去")
    if st.session_state.eeg_data is None or st.session_state.eval_data is None:
        st.warning("XDFファイルと試行情報ファイルの両方をアップロードしてください。")
        return

    # 特徴量計算
    if st.button("📈 1秒ごとの特徴量を計算", type="primary"):
        with st.spinner("特徴量を計算中..."):
            filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
            # 1秒ごとの特徴量を計算（画像全体を対象とするためtime_rangeは固定）
            # ここでは例として-1秒から10秒の試行を対象とする
            features_df = calculate_features(filtered_eeg, st.session_state.eval_data, (-1.0, 10.0))
            st.session_state.raw_features_df = features_df
            st.session_state.filtered_features_df = features_df
            st.success(f"{len(features_df)}個のデータポイント（1秒毎）が生成されました。")

    if st.session_state.raw_features_df is None:
        st.info("上のボタンを押して、特徴量計算を開始してください。")
        return
        
    st.markdown("---")
    st.subheader("📊 散布図による外れ値の可視化")
    
    df = st.session_state.raw_features_df
    feature_cols = [col for col in df.columns if col not in ['img_id', 'second']]
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X軸", feature_cols, index=0)
    with col2:
        y_axis = st.selectbox("Y軸", feature_cols, index=1 if len(feature_cols) > 1 else 0)

    # 閾値設定UI
    st.markdown("##### 除去する閾値を設定")
    col1, col2, _, col3 = st.columns([2, 2, 1, 1])
    with col1:
        x_thresh = st.number_input(f"X軸 ({x_axis}) の上限値", value=df[x_axis].quantile(0.99))
    with col2:
        y_thresh = st.number_input(f"Y軸 ({y_axis}) の上限値", value=df[y_axis].quantile(0.99))
    with col3:
        st.write("") # スペース
        if st.button("閾値を適用"):
            filtered_df = df[(df[x_axis] < x_thresh) & (df[y_axis] < y_thresh)]
            st.session_state.filtered_features_df = filtered_df

    # フィルタリング結果の表示
    filtered_df = st.session_state.filtered_features_df
    original_count = len(df)
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("元の点数", original_count)
    col2.metric("除去された点数", removed_count, delta=-removed_count)
    col3.metric("残りの点数", filtered_count)

    # 散布図描画
    fig = plot_outlier_scatter(df, x_axis, y_axis, x_thresh, y_thresh)
    st.info("このプロットは常に**全ての点**を表示し、赤線で閾値を示します。")
    st.plotly_chart(fig, use_container_width=True)
    
    # フィルタリング後のデータテーブル
    st.markdown("---")
    st.subheader("📋 閾値で除去した後のデータ")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 除去後データをCSVでダウンロード", csv, "filtered_features.csv", "text/csv")


# --- メイン実行部 ---
def main():
    check_password()
    initialize_session_state()
    st.title("🧠 EEG Analysis App")
    controls = sidebar_controls()
    # タブ構成を変更
    outlier_rejection_tab(controls)

if __name__ == "__main__":
    main()
