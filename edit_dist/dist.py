#!/usr/bin/python

from __future__ import print_function
import sys

def needleman_wunsch(s1,s2,penalty_func,null_char):
    def get_alignment(matrix):
        ext1 = ''
        ext2 = ''

        (i,j) = (len(s1)-1,len(s2)-1)
        while i >= 0 or j >= 0:
            if i > 0 and \
                    matrix[(i,j)] == (matrix[(i-1,j)] + penalty_func(s1[i],null_char)):
                ext1 = s1[i] + ext1
                ext2 = null_char + ext2
                i -= 1
            elif j > 0 and \
                    matrix[(i,j)] == (matrix[(i,j-1)] + penalty_func(s2[j],null_char)):
                ext1 = null_char + ext1
                ext2 = s2[j] + ext2
                j -= 1
            else:
                ext1 = s1[i] + ext1
                ext2 = s2[j] + ext2
                i -= 1
                j -= 1

        return (ext1,ext2)

    matrix = {}
    for i,x in enumerate(s1):
        matrix[(i,-1)] = penalty_func(x,null_char)
    for j,x in enumerate(s2):
        matrix[(-1,j)] = penalty_func(x,null_char)
    matrix[(-1,-1)] = 0
    for i,c1 in enumerate(s1):
        for j,c2 in enumerate(s2):
            match = matrix[(i-1,j-1)] + penalty_func(c1,c2)
            delete = matrix[(i-1,j)] + penalty_func(c1,null_char)
            insert = matrix[(i,j-1)] + penalty_func(c2,null_char)
            matrix[(i,j)] = min(match, delete, insert)

    return (matrix[(len(s1)-1,len(s2)-1)], get_alignment(matrix))

null_char = ' '
def penalty_func(c1,c2):
    if c1 == c2: return 0
    elif c1 == null_char or c2 == null_char: return 1
    else: return 0.5

if len(sys.argv) < 3:
    print('usage error',file=sys.stderr)
    sys.exit(1)

(cost,(ext1,ext2)) = needleman_wunsch(sys.argv[1],sys.argv[2],penalty_func,null_char)
print('Cost: ' + str(cost))
print(ext1)
print(ext2)
