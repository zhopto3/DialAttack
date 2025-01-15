"""A script that takes as input the split and the sample number of interest and then outputs the relevant demographic information:
Client_ids: The average number of recordings per client id
Gender: the proportion of recordings by females
Age: The average age
"""
import argparse
import pandas as pd
import numpy as np
import csv

AGES = {
    'teens':15,
    'twenties':25,
    'thirties':35,
    'fourties':45,
    'fifties':55,
    'sixties':65,
    'seventies':75,
    'eighties':85,
    'nineties':95
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", choices=['train','dev'], default='test', required=True)
    parser.add_argument("--sample", type=int, choices = [1,2,3,4,5], required = True)
    parser.add_argument('--prop_central',type=int, choices=[20,50,80,100], required=True)

    return parser.parse_args()


def get_descriptives(args, df):
    #first the average number of contributions per client id 
    avg_contributions = df['client_id'].value_counts().sum()/df['client_id'].nunique()
    #The number of unique contributers/clients ids, normalized by the total number of people
    normalized_unique = df['client_id'].nunique()/len(df['client_id'])
    #print(df['client_id'].nunique(), len(df['client_id']))

    #proportion female speakers in all the saples
    prop_fem = len(df[df['gender']=='female_feminine'])/len(df['gender'])
    
    #Average age (assuming the mid point for the provided age range)
    avg_age = sum([df['age'].value_counts()[age]*val if age in df['age'].unique() else 0 for age,val in list(AGES.items())])/len(df['age'])

    return args.split, args.prop_central, args.sample, avg_contributions, normalized_unique, prop_fem, avg_age
    

if __name__=="__main__":
    args = get_args()
    #get the relevant data set
    df =pd.read_csv(f'./samples/samp_0{args.sample}/{args.split}_{args.prop_central}.tsv',delimiter="\t", escapechar="\\",quoting = csv.QUOTE_NONE)

    print(*get_descriptives(args, df),sep='\t')