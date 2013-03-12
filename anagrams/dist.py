#!/usr/bin/python

from __future__ import print_function
import sys

def needleman_wunsch(s1,s2,penalty_func,null_char):
    matrix = {}
    matrix[(-1,-1)] = 0
    for i,x in enumerate(s1):
        matrix[(i,-1)] = matrix[(i-1,-1)] + penalty_func(x,null_char)
    for j,x in enumerate(s2):
        matrix[(-1,j)] = matrix[(-1,j-1)] + penalty_func(x,null_char)
    for i,c1 in enumerate(s1):
        for j,c2 in enumerate(s2):
            match = matrix[(i-1,j-1)] + penalty_func(c1,c2)
            delete = matrix[(i-1,j)] + penalty_func(c1,null_char)
            insert = matrix[(i,j-1)] + penalty_func(c2,null_char)
            matrix[(i,j)] = min(match, delete, insert)

    return matrix[(len(s1)-1,len(s2)-1)]

def simple_dist(s1,s2):
    def penalty_func(c1,c2):
        if c1 == c2: return 0
        else: return 1

    return needleman_wunsch(s1,s2,penalty_func,' ')
