import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_waveforms(plot_data, display_mode="重ねて"):
    """生波形とフィルター後波形をプロット（変更なし）"""
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

def plot_outlier_scatter(data, x_col, y_col, x_thresh=None, y_thresh=None):
    """
    外れ値検出のための散布図を描画する
    """
    if data.empty:
        return go.Figure()

    fig = px.scatter(data, x=x_col, y=y_col,
                     hover_data=['img_id', 'second'],
                     title=f"<b>{x_col}</b> vs <b>{y_col}</b>")
    
    # 閾値の線を追加
    if x_thresh is not None:
        fig.add_vline(x=x_thresh, line_dash="dash", line_color="red", annotation_text="X Threshold")
    if y_thresh is not None:
        fig.add_hline(y=y_thresh, line_dash="dash", line_color="red", annotation_text="Y Threshold")
    
    fig.update_layout(template="plotly_white", height=500)
    return fig
