#!/bin/bash
# エラーが発生したらそこで処理を停止する
set -e

echo "=== 1. Running Initial Functions ==="
python -u initial_function_eng.py
python -u initial_function_jp.py
python -u initial_function_math.py
python -u initial_function_grades.py

echo "=== 2. Starting Main Processor ==="
# exec をつけることで、メインプロセスとして実行されコンテナの停止シグナルなどを正しく受け取れるようになります
exec python -u pdf_knowledge_processor.py