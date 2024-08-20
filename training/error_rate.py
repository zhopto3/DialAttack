""""A script that uses dynamic programming to calculate either the normalized word error rate or character error rate between two seq"""

import numpy as np


def error_rate(gold, model, char):
    if char:
        gold_toks = list(gold)
        hyp_toks = list(model)

    else:
        gold_toks = gold.split()
        hyp_toks = model.split()
    
    #set up matrix: 
    mat = np.zeros((len(gold_toks)+1,len(hyp_toks)+1), dtype=int)
    #The first row/column represent the distance between each string and an empty string; fill them to reflect that
    mat[:,0] = np.array(range(len(gold_toks)+1))
    mat[0] = np.array(range(len(hyp_toks)+1))

    #now fill table, starting from second row/column
    for i in range(1,len(gold_toks)+1):
        for j in range(1,len(hyp_toks)+1):
            #Check if the word at the prev indices are equal to determine the local cost at a given cell
            cost = 0 if gold_toks[i-1] == hyp_toks[j-1] else 1
            #Check the prior three cells (horizontally, diagonally, and vertically) to select the minimum global cost for the cell
            mat[i][j]= cost + min(mat[i-1][j-1],mat[i][j-1],mat[i-1][j])
    min_cost = mat[len(gold_toks),len(hyp_toks)]

    #Return the cost normalized by the number of tokens in the reference
    return min_cost/len(gold_toks)