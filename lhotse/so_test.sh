#!/bin/bash

base_dir="/data/duhu/wenet/examples/ct_sim/s0/data"
exp_dir="/data/duhu/k2/icefall/egs/ct_gz/ASR/data/manifest/subset_test/"
for folder in `ls $base_dir | grep -v "done"`; do 
    cur_dir=$base_dir/$folder
    if [ ! -f $cur_dir/test/wav.scp ]; then 
        continue
    fi 
    if [ ! -f $cur_dir/test/segments ]; then 
        continue 
    fi 
    if [ ! -f $cur_dir/test/text ]; then 
        continue
    fi
    # echo $cur_dir 
    dir_name=`basename $cur_dir` 
    if [ -f $exp_dir/.$dir_name.done ]; then
        # echo "$dir_name already done..."
        continue
    fi
    echo -n $dir_name" "
    # lhotse prepare so-kaldi $cur_dir/test $exp_dir -j 32 -s $dir_name 
done
