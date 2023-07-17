#!/bin/bash

# start simu_len
start_len=50

# end simu_len
end_len=190

# increment step
increment=10

for (( simu_len=$start_len; simu_len<=$end_len; simu_len+=$increment ))
do
    python launcher.py --exp_type=test --simu_len=$simu_len --num_ex=20 --pg_re=data/pg_re_2280.pkl --unseen=True
done
