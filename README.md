# EEGcheck
2チャンネルEEG (Fp1, Fp2) のXDFデータと、同一トライアルの主観評価データを入力し、インタラクティブな波形比較と周波数解析を行うStreamlitアプリケーションです。

## ✨ 主な機能

- **ファイルアップロード**:
  - EEGデータ (`.xdf`形式)
  - 主観評価データ (`.csv` or `.xlsx`形式)
- **インタラクティブな波形ビューア**:
  - 生データとフィルター後データの比較（重ねて/並べて表示）
  - 表示範囲を画像IDまたは秒数で指定可能
- **周波数解析とEDA**:
  - Welch法によるパワースペクトル密度 (PSD) 特徴量の算出
    - 各バンドパワー（α, β, θ）と相対パワー
    - 前頭葉非対称性 (Frontal Asymmetry)
    - ピークアルファ周波数
  - EEG特徴量と主観評価の散布図表示（回帰直線付き）
  - ピアソン相関係数とp値の自動計算
  - 解析結果のテーブル表示とCSVダウンロード機能

## ⚙️ ローカルでの実行方法

1.  **リポジトリをクローン:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **必要なライブラリをインストール:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **環境変数を設定（任意）:**
    ローカルでパスワード認証をテストする場合、環境変数を設定します。
    ```bash
    # for Linux/macOS
    export APP_PASSWORD="your_local_password"
    
    # for Windows (Command Prompt)
    set APP_PASSWORD="your_local_password"
    ```
    設定しない場合、デフォルトのパスワード `eeg2024` が使用されます。

4.  **Streamlitアプリを実行:**
    ```bash
    streamlit run app.py
    ```
    ブラウザで `http://localhost:8501` が開きます。

## ☁️ Streamlit Community Cloudへのデプロイ

1.  **GitHubリポジトリの準備:**
    - このプロジェクトの全ファイル (`app.py`, `loader.py`, `preprocess.py`, `features.py`, `utils_plot.py`, `requirements.txt`, `README.md`) をGitHubリポジトリにプッシュします。

2.  **Streamlit Cloudにデプロイ:**
    - [Streamlit Community Cloud](https://share.streamlit.io/)にログインします。
    - "New app" ボタンをクリックし、"Deploy from GitHub" を選択します。
    - 作成したリポジトリ、ブランチ（例: `main`）、メインファイル（`app.py`）を選択します。

3.  **パスワード認証の設定 (Secrets):**
    - "Advanced settings..." をクリックします。
    - **Secrets** のテキストボックスに、アプリのパスワードをTOML形式で入力します。

      ```toml
      # .streamlit/secrets.toml
      APP_PASSWORD = "your_secure_password_here"
      ```

    - "Save" をクリックします。

4.  **デプロイ実行:**
    - "Deploy!" ボタンをクリックします。
    - デプロイが完了すると、あなたのアプリが公開されます。

## 📝 注意事項

- **XDFファイル**: `Fp1`, `Fp2` というラベルを持つチャンネルと、画像IDを値として持つマーカーストリームが必要です。サンプリング周波数は250Hz以上を推奨します。
- **評価データ**: `img_id` 列が必須です。この列をキーにしてEEGのマーカーと紐付けられます。
- **パフォーマンス**: 大量のトライアルを含むデータを解析する場合、特徴量計算に時間がかかることがあります。計算中はスピナーが表示されます。
