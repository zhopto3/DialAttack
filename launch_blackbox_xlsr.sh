#!/bin/bash

mkdir ./experiments/central20_53m_v01/xlsr_robust
mkdir ./experiments/central50_53m_v01/xlsr_robust
mkdir ./experiments/central80_53m_v01/xlsr_robust
mkdir ./experiments/central100_53m_v02/xlsr_robust

#attack 100% Central
python3 ./attack/blackbox_xlsr.py --attack_experiment 80_53m_v01 --experiment_subject central100_53m_v02 --model XLSR53 > ./experiments/central100_53m_v02/xlsr_robust/central80_attack100.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 50_53m_v01 --experiment_subject central100_53m_v02 --model XLSR53 > ./experiments/central100_53m_v02/xlsr_robust/central50_attack100.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 20_53m_v01 --experiment_subject central100_53m_v02 --model XLSR53 > ./experiments/central100_53m_v02/xlsr_robust/central20_attack100.tsv

#attack 80% Central
python3 ./attack/blackbox_xlsr.py --attack_experiment 100_53m_v02 --experiment_subject central80_53m_v01 --model XLSR53 > ./experiments/central80_53m_v01/xlsr_robust/central100_attack80.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 50_53m_v01 --experiment_subject central80_53m_v01 --model XLSR53 > ./experiments/central80_53m_v01/xlsr_robust/central50_attack80.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 20_53m_v01 --experiment_subject central80_53m_v01 --model XLSR53 > ./experiments/central80_53m_v01/xlsr_robust/central20_attack80.tsv

#attack 50% Central
python3 ./attack/blackbox_xlsr.py --attack_experiment 100_53m_v02 --experiment_subject central50_53m_v01 --model XLSR53 > ./experiments/central50_53m_v01/xlsr_robust/central100_attack50.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 80_53m_v01 --experiment_subject central50_53m_v01 --model XLSR53 > ./experiments/central50_53m_v01/xlsr_robust/central80_attack50.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 20_53m_v01 --experiment_subject central50_53m_v01 --model XLSR53 > ./experiments/central50_53m_v01/xlsr_robust/central20_attack50.tsv

#attack 20% Central
python3 ./attack/blackbox_xlsr.py --attack_experiment 100_53m_v02 --experiment_subject central20_53m_v01 --model XLSR53 > ./experiments/central20_53m_v01/xlsr_robust/central100_attack20.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 80_53m_v01 --experiment_subject central20_53m_v01 --model XLSR53 > ./experiments/central20_53m_v01/xlsr_robust/central80_attack20.tsv
python3 ./attack/blackbox_xlsr.py --attack_experiment 50_53m_v01 --experiment_subject central20_53m_v01 --model XLSR53 > ./experiments/central20_53m_v01/xlsr_robust/central50_attack20.tsv
