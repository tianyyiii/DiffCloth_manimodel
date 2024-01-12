#!/bin/bash

MAX_PROCESSES=1
NAMES_FILE=$1
PARAM2=$2
PARAM3=$3

current_processes=0

while read -r name; do
    # 启动新的后台进程，并将输出重定向到日志文件
    /bin/python3 ./src/python_code/start.py "$name" "$PARAM2" "$PARAM3" > "./logs/${name}.log" 2>&1 &
    ((current_processes++))

    # 如果活跃进程数达到最大值，则等待一个进程完成
    if [[ $current_processes -ge $MAX_PROCESSES ]]; then
        wait -n
        ((current_processes--))
    fi
done < "$NAMES_FILE"

wait
