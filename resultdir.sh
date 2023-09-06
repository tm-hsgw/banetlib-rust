#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <args>"
  exit 1
fi

# 引数を取得
args="$1"

# 現在の日時からmmddhhmm形式の文字列を生成
timestamp=$(date +"%m%d%H%M")

# ディレクトリ名を生成
dirname="res/${timestamp}_${args}"

# ディレクトリを作成
mkdir -p "$dirname"

cd "$dirname"

touch "${args}.csv"
