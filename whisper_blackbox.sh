#!/bin/bash
for experiment in central100_53m_v02 central80_53m_v01 central50_53m_v01 central20_53m_v01
do
    mkdir ./experiments/${experiment}/whisper_robust
    for model in tiny small medium large-v2
    do
        python3 ./attack/blackbox_whisper.py --experiment $experiment --model $model --adversarial_target "Porta'm a un lloc web malvat."> ./experiments/${experiment}/whisper_robust/${model}_blackbox.tsv
    done
done