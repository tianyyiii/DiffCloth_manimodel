#!/bin/bash

# 设置最大线程数
MAX_THREAD=16

# 输出文件夹路径
OUTPUT_PATH="./output"

# 确保输出目录存在
mkdir -p $OUTPUT_PATH

# 生成参数组合并使用 xargs 并行运行 Python 脚本
for a in {0..1}; do
    for ep in {1..6}; do
        for step in {0..9}; do
            echo $a $ep $step $OUTPUT_PATH
        done
    done
done | xargs -n 4 -P $MAX_THREAD python3 src/python_code/process_tie_data.py
