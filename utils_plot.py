import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_waveforms(epoch_data, display_mode="重ねて", outlier_df=None):
    """
    生波形とフィルター後波形をプロットする。
    outlier_dfが指定された場合、除去された区間をハイライトする。
    """
    raw, filtered, times = epoch_data['raw'], epoch_data['filtered'], epoch_data['times']
    ch_names = ['Fp1', 'Fp2']
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=ch_names, vertical_spacing=0.1)

    for i, ch in enumerate(ch_names):
        fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', legendgroup=ch, line_color=colors[i], opacity=0.4, showlegend=(i==0)), row=i+1 if display_mode=="並べて" else 1, col=1)
        fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch} (フィルター後)', legendgroup=ch, line_color=colors[i], showlegend=(i==0)), row=i+1 if display_mode=="並べて" else 1, col=1)
    
    if outlier_df is not None and not outlier_df.empty:
        for _, row in outlier_df.iterrows():
            start_sec = row['second']
            end_sec = row['second_end']
            marker_relative_start = start_sec + epoch_data['time_range'][0]
            marker_relative_end = end_sec + epoch_data['time_range'][0]
            fig.add_vrect(x0=marker_relative_start, x1=marker_relative_end, fillcolor="rgba(255, 100, 100, 0.2)", layer="below", line_width=0)

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Marker")
    fig.update_layout(height=500, template="plotly_white", hovermode="x unified", legend_orientation="h")
    if display_mode == "重ねて":
        fig.update_layout(title="EEG波形比較（重ねて表示）")
        fig.update_yaxes(title_text="振幅 (μV)", row=1, col=1)
    else:
        fig.update_layout(title="EEG波形比較（並べて表示）")
        fig.update_yaxes(title_text="振幅 (μV)")
    return fig

def plot_outlier_scatter(data, x_col, y_col, x_thresh=None, y_thresh=None):
    """
    外れ値検出のための散布図を描画する（堅牢版）
    """
    # ★★ ここを修正 ★★
    # 探す列名を 'second' から 'window_start_sec' に変更
    required_cols = [x_col, y_col, 'img_id', 'window_start_sec']
    if not all(col in data.columns for col in required_cols):
        return go.Figure().update_layout(title=f"エラー: 必要な列が見つかりません {required_cols}")

    clean_data = data.dropna(subset=[x_col, y_col])
    if clean_data.empty:
        return go.Figure().update_layout(title="プロット可能なデータがありません")

    # ★★ ここを修正 ★★
    # hover_dataも 'second' から 'window_start_sec' に変更
    fig = px.scatter(clean_data, x=x_col, y=y_col,
                     hover_data=['img_id', 'window_start_sec'],
                     title=f"<b>{x_col}</b> vs <b>{y_col}</b>")
    
    if x_thresh is not None:
        fig.add_vline(x=x_thresh, line_dash="dash", line_color="red", annotation_text="X Threshold")
    if y_thresh is not None:
        fig.add_hline(y=y_thresh, line_dash="dash", line_color="red", annotation_text="Y Threshold")
    
    fig.update_layout(template="plotly_white", height=500)
    return fig
