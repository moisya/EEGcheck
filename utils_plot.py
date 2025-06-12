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
        # 重ねて表示でも並べて表示でも対応できるように、常にsubplotを使う
        if display_mode == "重ねて":
            fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', legendgroup=ch, line_color=colors[i], opacity=0.4, showlegend=(i==0)), row=1, col=1)
            fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch} (フィルター後)', legendgroup=ch, line_color=colors[i], showlegend=(i==0)), row=1, col=1)
        else: # 並べて
            fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', legendgroup=ch, line_color=colors[i], opacity=0.4, showlegend=(i==0)), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch} (フィルター後)', legendgroup=ch, line_color=colors[i], showlegend=(i==0)), row=i+1, col=1)
    
    # 外れ値区間をハイライト
    if outlier_df is not None and not outlier_df.empty:
        for _, row in outlier_df.iterrows():
            sec_start = row['second']
            sec_end = sec_start + 1
            # マーカーからの相対時間に変換
            marker_relative_start = sec_start + epoch_data['time_range'][0]
            marker_relative_end = sec_end + epoch_data['time_range'][0]
            
            fig.add_vrect(x0=marker_relative_start, x1=marker_relative_end, 
                          fillcolor="rgba(255, 100, 100, 0.2)", layer="below", line_width=0,
                          annotation_text="Removed", annotation_position="top left")

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Marker")
    fig.update_layout(height=500, template="plotly_white", hovermode="x unified", legend_orientation="h")
    if display_mode == "重ねて":
        fig.update_layout(title="EEG波形比較（重ねて表示）")
        fig.update_yaxes(title_text="振幅 (μV)", row=1, col=1)
    else:
        fig.update_layout(title="EEG波形比較（並べて表示）")
        fig.update_yaxes(title_text="振幅 (μV)")

    return fig

def plot_outlier_scatter(data, x_col, y_col, color_col, x_thresh=None, y_thresh=None):
    """
    外れ値検出のための散布図を、主観評価で色分けして描画する
    """
    if data.empty or color_col not in data.columns:
        return go.Figure().update_layout(title="データ不足または列が見つかりません")

    clean_data = data.dropna(subset=[x_col, y_col, color_col])
    
    fig = px.scatter(clean_data, x=x_col, y=y_col,
                     color=color_col,
                     hover_data=['img_id', 'second'],
                     title=f"<b>{x_col}</b> vs <b>{y_col}</b> (色: {color_col})")
    
    # 閾値の線を追加
    if x_thresh is not None:
        fig.add_vline(x=x_thresh, line_dash="dash", line_color="red", annotation_text="X Threshold")
    if y_thresh is not None:
        fig.add_hline(y=y_thresh, line_dash="dash", line_color="red", annotation_text="Y Threshold")
    
    fig.update_layout(template="plotly_white", height=600)
    return fig
