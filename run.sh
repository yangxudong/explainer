#!/bin/bash
input_file=$1
num_line=(`wc -l $input_file`)
avg_line=`expr ${num_line[0]} / 24 + 1`
split -d -l $avg_line $input_file part

for file in $(ls part*)
do {
  ./keyword_mine.py --input $file --output ${file}_output
} &
done
wait

