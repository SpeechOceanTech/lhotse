#!/bin/bash

base_dir="/data/duhu/wenet/examples/zh_cn/s0/data"
exp_dir="/data/duhu/k2/icefall/egs/zh_cn_13w/ASR/data/manifest/subset/"
for folder in `ls $base_dir | grep -v "done"`; do 
    cur_dir=$base_dir/$folder
    if [ ! -f $cur_dir/train/wav.scp ]; then 
        continue
    fi 
    if [ ! -f $cur_dir/train/segments ]; then 
        continue 
    fi 
    if [ ! -f $cur_dir/train/text ]; then 
        continue
    fi
    echo $cur_dir 
    dir_name=`basename $cur_dir` 
    if [ -f $exp_dir/.$dir_name.done ]; then
        echo "$dir_name already done..."
        continue
    fi
    lhotse prepare so-kaldi $cur_dir/train /data/duhu/k2/icefall/egs/zh_cn_13w/ASR/data/manifest/subset/ -j 32 -s $dir_name 
done
