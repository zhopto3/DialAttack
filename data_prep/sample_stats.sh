#!/bin/bash


echo -e "split\tprop_central\tsample\tavg_contributions\tunique_contributors\tproportion_female\tavg_age" > ./data_analysis/sample_demographics.tsv
for samp in 1 2 3 4 5
do
    for prop in 20 50 80 100
    do
        python3 ./data_prep/demographics.py --split "dev" --samp $samp --prop_central $prop >> ./data_analysis/sample_demographics.tsv
        python3 ./data_prep/demographics.py --split "train" --samp $samp --prop_central $prop >> ./data_analysis/sample_demographics.tsv
    done
done