# 軽量なPython 3.9の環境をベースにする
FROM python:3.9-slim

# PDFを画像に変換するためのOSツール(poppler)をインストール
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# 作業フォルダを設定
WORKDIR /app

# 先ほど作った requirements.txt をコピーしてライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードやデータをすべてコンテナ内にコピー
COPY . .

# コンテナが起動したときにメインコードを実行する
CMD ["python", "pdf_knowledge_processor.py"]