import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import plotly.express as px

def _merge_overlapping_intervals(df):
    """重なり合う時間区間を統合するヘルパー関数"""
    if df.empty: return []
    df = df.sort_values('second').reset_index(drop=True)
    merged = []
    if len(df) > 0:
        current_start, current_end = df.loc[0, 'second'], df.loc[0, 'second_end']
        for i in range(1, len(df)):
            next_start, next_end = df.loc[i, 'second'], df.loc[i, 'second_end']
            if next_start < current_end:
                current_end = max(current_end, next_end)
            else:
                merged.append({'start': current_start, 'end': current_end})
                current_start, current_end = next_start, next_end
        merged.append({'start': current_start, 'end': current_end})
    return merged

def plot_waveforms(epoch_data, display_mode="重ねて", outlier_df=None):
    """生波形とフィルター後波形をプロットし、除去区間をハイライトする"""
    raw, filtered, times = epoch_data['raw'], epoch_data['filtered'], epoch_data['times']
    ch_names = ['Fp1', 'Fp2']
    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=ch_names, vertical_spacing=0.1)

    for i, ch in enumerate(ch_names):
        fig.add_trace(go.Scatter(x=times, y=raw[i], mode='lines', name=f'{ch} (生)', legendgroup=ch, line_color=colors[i], opacity=0.4, showlegend=(i==0)), row=i+1 if display_mode=="並べて" else 1, col=1)
        fig.add_trace(go.Scatter(x=times, y=filtered[i], mode='lines', name=f'{ch} (フィルター後)', legendgroup=ch, line_color=colors[i], showlegend=(i==0)), row=i+1 if display_mode=="並べて" else 1, col=1)
    
    if outlier_df is not None and not outlier_df.empty:
        merged_intervals = _merge_overlapping_intervals(outlier_df)
        for interval in merged_intervals:
            start_sec, end_sec = interval['start'], interval['end']
            marker_relative_start = start_sec + epoch_data['time_range'][0]
            marker_relative_end = end_sec + epoch_data['time_range'][0]
            fig.add_vrect(x0=marker_relative_start, x1=marker_relative_end, fillcolor="rgba(255, 100, 100, 0.25)", layer="below", line_width=0)

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Marker")
    fig.update_layout(height=500, template="plotly_white", hovermode="x unified", legend_orientation="h")
    title_text = "EEG波形比較（重ねて表示）" if display_mode == "重ねて" else "EEG波形比較（並べて表示）"
    fig.update_layout(title=title_text)
    fig.update_yaxes(title_text="振幅 (μV)")
    return fig

# ★★ ここを修正 ★★
def plot_outlier_scatter(data, x_col, y_col, color_col, x_thresh=None, y_thresh=None):
    """
    外れ値検出のための散布図を、指定された列で色分けして描画する
    """
    required_cols = [x_col, y_col, color_col, 'img_id', 'window_start_sec']
    if not all(col in data.columns for col in required_cols):
        missing = [c for c in required_cols if c not in data.columns]
        return go.Figure().update_layout(title=f"エラー: 必要な列が見つかりません {missing}")

    clean_data = data.dropna(subset=[x_col, y_col, color_col])
    if clean_data.empty: return go.Figure().update_layout(title="プロット可能なデータがありません")

    fig = px.scatter(clean_data, x=x_col, y=y_col,
                     color=color_col,  # ← 色分けの列を指定
                     hover_data=['img_id', 'window_start_sec'],
                     title=f"<b>{x_col}</b> vs <b>{y_col}</b> (色: {color_col})")
    
    if x_thresh is not None: fig.add_vline(x=x_thresh, line_dash="dash", line_color="red")
    if y_thresh is not None: fig.add_hline(y=y_thresh, line_dash="dash", line_color="red")
    
    fig.update_layout(template="plotly_white", height=500)
    return fig
