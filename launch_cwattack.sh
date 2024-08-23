#!/bin/bash
python3 ./attack/launch_attack.py --experiment_name central100_53m_v02 --model XLSR53 --lr 0.01 --regularizing_const 1.0 > ./experiments/central100_53m_v02/eval/attack_stats.tsv
python3 ./attack/launch_attack.py --experiment_name central80_53m_v01 --model XLSR53 --lr 0.01 --regularizing_const 1.0 > ./experiments/central80_53m_v01/eval/attack_stats.tsv
python3 ./attack/launch_attack.py --experiment_name central50_53m_v01 --model XLSR53 --lr 0.01 --regularizing_const 1.0 > ./experiments/central50_53m_v01/eval/attack_stats.tsv
python3 ./attack/launch_attack.py --experiment_name central20_53m_v01 --model XLSR53 --lr 0.01 --regularizing_const 1.0 > ./experiments/central20_53m_v01/eval/attack_stats.tsv