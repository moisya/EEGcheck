import os
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import pyxdf
import streamlit as st


# ────────────────────────────────────────────────
# XDF ─────────────────────────────────────────────
# ────────────────────────────────────────────────
@st.cache_data(show_spinner="XDF ファイルを解析しています…")
def load_xdf_data(uploaded_file):
    """
    XDF ファイルから 2ch EEG (Fp1, Fp2) とマーカーを抽出して辞書で返す。
    返り値: {"eeg_stream": {...}, "markers": DataFrame}
    """
    try:
        # ① 一時ファイルに保存（pyxdf は file-path しか受けない）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # ② 読み込み
        streams, _ = pyxdf.load_xdf(tmp_path, verbose=False)
    finally:
        # ③ 後始末
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    eeg_stream, marker_df = None, None

    # ④ ストリーム特定
    for s in streams:
        info = s["info"]

        # ── EEG ストリーム判定 ──────────────────────
        #   ・type が "EEG" など
        #   ・チャンネルに Fp1, Fp2 が含まれる
        stype = info["type"][0].lower()
        if "eeg" in stype:
            # channel ラベル取得（XML 構造は実装差があるため try/except）
            try:
                ch_nodes = info["desc"][0]["channels"][0]["channel"]
                labels = [ch["label"][0] for ch in ch_nodes]
            except (KeyError, IndexError, TypeError):
                labels = []

            if {"Fp1", "Fp2"} <= set(labels):
                sfreq = float(info["nominal_srate"][0])
                if sfreq < 250:
                    st.error(f"サンプリングレートが {sfreq} Hz です。250 Hz 以上が必要です。")
                    return None

                fp1_idx, fp2_idx = labels.index("Fp1"), labels.index("Fp2")
                eeg_stream = {
                    "data": s["time_series"][:, [fp1_idx, fp2_idx]].T,  # shape (2, n_samples)
                    "times": s["time_stamps"],                          # shape (n_samples,)
                    "sfreq": sfreq,
                    "ch_names": ["Fp1", "Fp2"],
                }

        # ── Marker ストリーム判定 ──────────────────
        if stype in {"markers", "marker", "stim"} or (
            "marker" in info["name"][0].lower()
        ):
            marker_rows = []
            for ts, ts_values in zip(s["time_stamps"], s["time_series"]):
                if not ts_values:
                    continue
                try:
                    m_val = int(str(ts_values[0]).strip())
                    marker_rows.append({"marker_time": ts, "img_id": m_val})
                except ValueError:
                    # 数字にならないマーカーは無視
                    continue

            if marker_rows:
                marker_df = pd.DataFrame(marker_rows, dtype="float64")

    # ⑤ 存在チェック
    if eeg_stream is None:
        st.error("Fp1 / Fp2 を含む EEG ストリームが見つかりませんでした。")
        return None

    if marker_df is None:
        st.warning("整数マーカーが見つかりませんでした。画像 ID 指定は利用不可です。")
        marker_df = pd.DataFrame(columns=["marker_time", "img_id"])

    st.success(f"EEG 読み込み完了 ({eeg_stream['sfreq']} Hz, "
               f"{eeg_stream['data'].shape[1]:,} samples)")
    return {"eeg_stream": eeg_stream, "markers": marker_df}


# ────────────────────────────────────────────────
# 評価 CSV / Excel ───────────────────────────────
# ────────────────────────────────────────────────
@st.cache_data(show_spinner="評価データを解析中…")
def load_evaluation_data(uploaded_file):
    """
    評価データ (CSV / XLSX) を DataFrame で返す。
    ・必須列: img_id
    ・数値列: Dislike_Like, sam_val, sam_aro を自動 numeric 変換
    """
    try:
        # ① 拡張子チェック
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("CSV か Excel(.xlsx/.xls) をアップロードしてください。")
            return None

        # ② 前処理
        df.columns = df.columns.str.strip()  # 空白除去
        if "img_id" not in df.columns:
            st.error("必須列 'img_id' が見つかりません。")
            return None

        # img_id → int へ　(欠損は行ごとドロップ)
        df["img_id"] = pd.to_numeric(df["img_id"], errors="coerce")
        df = df.dropna(subset=["img_id"]).reset_index(drop=True)
        df["img_id"] = df["img_id"].astype(int)

        # 評価列を numeric へ
        for col in ["Dislike_Like", "sam_val", "sam_aro"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.success(f"評価データ読み込み完了 ({len(df):,} rows)")
        return df

    except Exception as e:
        st.error(f"評価データの読み込みに失敗しました: {e}")
        return None
