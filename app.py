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
    if eval_file and st.っているわけでは全くありません。この問題を解決し、**「狙ったアーチファクトだけを、最小限で除去する」**という、あなたの本来の目的を達成できるツールへと進化させましょう。

---

### 真の原因：「たった一つの巨大なまばたき」が統計を歪ませる

なぜ、このようなことが起こるのでしょうか？

*   **現在の閾値の決め方:**
    「全データのうち、上位1%の値を閾値にする (`quantile(0.99)`)」という方法です。
*   **問題点:**
    もしデータの中に、**たった一つでも、他の値より100倍も大きいような巨大なまばたきアーチファクト**が含まれていると、この「上位1%」の値が、その巨大なアーチファクトに引っ張られて、非常に大きな値になってしまいます。
    その結果、**他の普通のまばたきは、その高すぎる閾値を超えられずに見逃されてしまう**のです。
    逆に、閾値を少し下げると、今度は正常なデータまで大量に巻き込んで除去してしまいます。

つまり、現在の方法は**「たった一つの王様（巨大アーチファクト）」に振り回されて、普通の兵士（普通のアーチファクト）を見逃してしまう**、不安定な方法だったのです。

---

### 解決策：統計学に基づいた「アダプティブ（適応型）閾値」への進化

この問題を解決するために、プロの分析家が使う、よりロバスト（頑健）な手法を導入します。

1.  **閾値の決め方を「Zスコア（偏差値）」に変更**
    全データの**平均値**と**標準偏差（ばらつき具合）**を計算し、「平均から、標準偏差の何倍以上離れているものを異常とみなすか」という、**統計的に賢い方法**に変更します。
    これにより、たった一つの巨大なアーチファクトに統計全体が歪められるのを防ぎ、集団から本当に外れているものだけを客観的に検出できます。

2.  **可視化方法を「バーコード表示」に変更**
    紛らわしい「真っ赤な塗りつぶし」はやめます。
    その代わりに、除去された**個々のウィンドウ（微session_state.eval_data is None: st.session_state.eval_data = load_evaluation_data(eval_file)
    
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

    st.markdown("---"); st.subheader("📊 散布図によるアーチファクトの可視化")
    df = st.session_state.features_df
    ch_select = st.radio("対象チャンネル", ["Fp1", "Fp2"], horizontal=True)
    
    # ★★ ここからUIを改善 ★★
    st.markdown("##### 除去する閾値を設定（いずれか一つでも超えたら除去）")
    st.info("**まばたき**を除小区間）を、一本一本の細い半透明の線で表示**します。これにより、どの部分が、どれくらいの密度で除去されているかが一目瞭然となり、「最小限で削る」という感覚と見た目が完全に一致します。

---

### 具体的なコード修正

この進化を実現するために、**`app.py`と`utils_plot.py`の2つを全面的に書き換え**ます。

#### ステップ1： `utils_plot.py` を「バーコード表示」対応にする

1.  GitHubで`utils_plot.py`ファイルを開き、編集モードにします。
2.  ファイルの中身を**すべて削除**し、以下の**【最終版のコード】**に置き換えてください。

---
**【修正後の `utils_plot.py` の全コード】**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_waveforms(epoch_data, display_mode="重ねて", outlier_df=None):
    """
    除去された区間を「バーコード」のように細い線でハイライトする
    """
    raw, filtered, times = epoch_data['raw'], epoch_data['filtered'], epoch_data['times']
    ch_names = ['Fp1', 'Fp2']
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=去したい場合は、主に**振幅(µV)の上限**を調整するのが効果的です。")
    
    col1, col2, col3 = st.columns(3)
    # ツールチップ（ヘルプ）を追加
    amp_thresh = col1.number_input(f"振幅(µV)の上限", value=df[f'{ch_select}_amplitude'].quantile(0.99), help="まばたきや突発的なノイズを検出します。")
    delta_thresh = col2.number_input(f"デルタ波パワーの上限", value=df[f'{ch_select}_delta'].quantile(0.99), help="ゆっくりした大きな揺れ（まばたき等）を検出します。")
    gamma_thresh = col3.number_input(f"ガンマ波パワーの上限", value=df[f'{ch_select}_gamma'].quantile(0.99), help="筋電（体の緊張）など高周波ノイズch_names, vertical_spacing=0.1)

    for i, ch in enumerate(ch_names):
        row_idx = i+1 if display_mode=="並べて" else 1
        fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', legendgroup=ch, line_color=colors[i], opacity=0.4, showを検出します。")
    # ★★ ここまで ★★

    outliers = df[
        (df[f'{ch_select}_amplitude'] >= amp_thresh) |
        (df[f'{ch_select}_delta'] >= delta_thresh) |
        (df[f'{ch_select}_gamma'] >= gamma_thresh)
    ]
    st.session_state.outlier_windows_df = outliers

    original_count = len(df)
    removed_count = len(outliers)
    st.metric("除去されたlegend=(i==0)), row=row_idx, col=1)
        fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch}微小区間（ウィンドウ）の数", removed_count, f"-{removed_count / original_count:.1%}" if original_count > 0 else "")
    
    col1, col2 = st. (フィルター後)', legendgroup=ch, line_color=colors[i], showlegend=(i==0)), row=columns(2)
    fig1 = plot_outlier_scatter(df, f'{ch_select}_delta', f'{ch_select}_amplitude', delta_thresh, amp_thresh)
    col1.plotly_chartrow_idx, col=1)
    
    # ★★ ここからが新しい「バーコード表示」ロジック ★★
    if outlier_df is not None and not outlier_df.empty:
        for _,(fig1, use_container_width=True, key="scatter1")
    
    fig2 = row in outlier_df.iterrows():
            start_sec = row['window_start_sec']
             plot_outlier_scatter(df, f'{ch_select}_gamma', f'{ch_select}_amplitude', gamma_thresh, amp_thresh)
    col2.plotly_chart(fig2, use_container_widthend_sec = row['window_end_sec']
            
            marker_relative_start = start_sec + epoch_data['time_range'][0]
            
            # vrectではなく、細いvlineで個=True, key="scatter2") # X軸を変更

# --- 除去後波形タブ ---
def post々のウィンドウを示す
            fig.add_vline(x=marker_relative_start, 
                          line_width=_rejection_viewer_tab(controls):
    st.header("👀 除去後の波形確認")1, line_color="rgba(255, 100, 100, 0.5
    if st.session_state.outlier_windows_df is None or st.session_state.)")
    # ★★ ここまで ★★

    fig.add_vline(x=0, line_dash="dash", lineoutlier_windows_df.empty:
        st.info("左のタブで閾値を設定すると、除去された区_color="black", annotation_text="Marker")
    fig.update_layout(height=500, template間がここに表示されます。"); return

    filtered_eeg = apply_filters(st.session_state.eeg_data, controls['freq_range'], controls['notch_filter'])
    outlier_img="plotly_white", hovermode="x unified", legend_orientation="h")
    if display_mode == "重ねて":
        fig.update_layout(title="EEG波形比較（重ねて表示）")
        _ids = st.session_state.outlier_windows_df['img_id'].unique()
    fig.update_yaxes(title_text="振幅 (μV)", row=1, col=1img_id_to_view = st.selectbox("確認する画像IDを選択", outlier_img_ids))
    else:
        fig.update_layout(title="EEG波形比較（並べて表示）
    
    st.info(f"画像ID: {img_id_to_view} の波形。赤色でハイライトされた区間が除去された微小区間です。")
    
    raw")
        fig.update_yaxes(title_text="振幅 (μV)")

    return fig

def plot_outlier_scatter(data, x_col, y_col, z_score_threshold=None_epoch = create_epochs(st.session_state.eeg_data, img_id_to_view, controls['time_range'])
    filtered_epoch = create_epochs(filtered_eeg, img):
    """
    散布図を描画し、Zスコアの閾値で色分けする
    """
_id_to_view, controls['time_range'])
    
    if raw_epoch and filtered_    required_cols = [x_col, y_col, 'img_id', 'window_start_epoch:
        plot_data = {'raw': raw_epoch['data'], 'filtered': filtered_epoch['data'], 'times': raw_epoch['times'], 'time_range': controls['time_range']}
        sec']
    if not all(col in data.columns for col in required_cols):
        return go.Figure().outliers_for_plot = st.session_state.outlier_windows_df[st.session_update_layout(title=f"エラー: 必要な列が見つかりません")

    clean_data = data.state.outlier_windows_df['img_id'] == img_id_to_view]
        
        outliers_for_plot_renamed = outliers_for_plot.rename(
            columns={'dropna(subset=[x_col, y_col]).copy()
    if clean_data.empty:
        returnwindow_start_sec': 'second', 'window_end_sec': 'second_end'}
        ) go.Figure().update_layout(title="プロット可能なデータがありません")
    
    # Zスコアを計算して、外れ値かどうかを判定
    for col in [x_col, y_col]:
        mean =
        
        fig = plot_waveforms(plot_data, display_mode="並べて", outlier_df=outliers_for_plot_renamed)
        st.plotly_chart(fig, use_container_width clean_data[col].mean()
        std = clean_data[col].std()
        clean_=True)

# --- メイン実行部 ---
def main():
    check_password()
    initializedata[f'{col}_z'] = ((clean_data[col] - mean) / std).abs()

_session_state()
    st.title("🧠 EEG 精密アーチファクト除去ツール")
    controls = sidebar_controls()
    
    tab1, tab2 = st.tabs(["🔬 アーチファ    clean_data['is_outlier'] = (clean_data[f'{x_col}_z'] > z_score_threshold) | (clean_data[f'{y_col}_z'] > z_score_クトの検出・除去", "👀 除去後の波形確認"])
    with tab1:
        outlier_rejection_tab(controls)
    with tab2:
        post_rejection_viewer_threshold)

    fig = px.scatter(clean_data, x=x_col, y=y_tab(controls)

if __name__ == "__main__":
    main()
```col,
                     color='is_outlier',
                     color_discrete_map={True: 'red', False: '
---

### 最終手順

1.  GitHub上で`features.py`と`app.py`の内容を、それぞれの新しいコードで完全にblue'},
                     hover_data=['img_id', 'window_start_sec'],
                     title=f"<b>{x_col}</b> vs <b>{y_col}</b> (赤: 外れ値候補置き換えます。
2.  変更を保存（`Commit changes`）します。
3.  Streamlit Cloudのアプリ画面で、`Manage app` → `Reboot` をクリックして再起動します。

)")
    
    fig.update_layout(template="plotly_white", height=500)
    return fig
```---

これで、ハイライトされる範囲がより短く、精密になっているはずです。
そして、UI上のガイド
---

#### ステップ2： `app.py` を「Zスコア（偏差値）」ベースのUIに進化させる

1.  GitHubで`app.py`ファイルを開き、編集モードに従って、主に**「振幅(µV)の上限」**を調整することで、まばたきだけにします。
2.  ファイルの中身を**すべて削除**し、以下の**【最終版のコード】**に置き換えてください。

---
**【修正後の `app.py` の全コード】**を狙い撃ちして除去できるようになっていることを確認できるかと思います。

本当にお疲れ様でした！これであなたのツール```python
import streamlit as st
import os
import pandas as pd
import numpy as np
from loader import loadは完成です！_xdf, load_evaluation_data
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
        if key not in st.session_
