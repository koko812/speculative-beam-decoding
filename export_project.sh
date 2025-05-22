#!/bin/bash

# 出力ファイル
OUTPUT="$HOME/Downloads/project_dump.txt"
> "$OUTPUT"

# 1. ツリー構造出力（.venv 除外）
echo "# 📁 Project Directory Tree" >> "$OUTPUT"
echo '```' >> "$OUTPUT"
tree -I '.venv|__pycache__' >> "$OUTPUT"
echo '```' >> "$OUTPUT"
echo "" >> "$OUTPUT"

# 2. 対象ファイル拡張子
EXTENSIONS=("py" "md" "yaml" "log" "jsonl" "json")

# 3. 各ファイルの内容を連結出力（.venv配下は除外）
for ext in "${EXTENSIONS[@]}"; do
  echo "# 📄 .$ext ファイル一覧" >> "$OUTPUT"
  echo "" >> "$OUTPUT"

  find . -type f -name "*.${ext}" ! -path "./.venv/*" | sort | while read -r file; do
    echo "## ▶️ ${file}" >> "$OUTPUT"
    echo '```'${ext} >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
  done
done

echo "✅ Export completed: $OUTPUT"

