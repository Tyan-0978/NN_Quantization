#!/usr/bin/bash

bits=$1
num_ops=$2

num_sessions=$3


activate="source env/bin/activate;"

from=1
to=50000
step=$((to/num_sessions))

for i in $(seq $from $step $to)
do
    j=$((i+step-1))
    tmux new-session -d "$activate python3 psum_exp.py $bits $num_ops $i $j"
done

tmux list-sessions
