#!/usr/bin/python

from __future__ import print_function
import sys

DEBUG = False

def needleman_wunsch(s1,s2,penalty_func,null_char):
    def get_alignment(matrix):
        ext1 = ''
        ext2 = ''

        (i,j) = (len(s1)-1,len(s2)-1)
        while i >= 0 or j >= 0:
            if i >= 0 and \
                (matrix[(i,j)] == (matrix[(i-1,j)] + penalty_func(s1[i],null_char)) \
                or j == -1):
                ext1 = s1[i] + ext1
                ext2 = null_char + ext2
                i -= 1
            elif j >= 0 and \
                (matrix[(i,j)] == (matrix[(i,j-1)] + penalty_func(s2[j],null_char)) \
                or i == -1):
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

    if DEBUG:
        for j in range(-1,len(s2)):
            print('\t'.join([str(matrix[(i,j)]) for i in range(-1,len(s1))]))

    return (matrix[(len(s1)-1,len(s2)-1)], get_alignment(matrix))

null_char = ' '
vowels = 'aeiou'
def penalty_func(c1,c2):
    if c1 == c2: return 0
    elif c1 == null_char or c2 == null_char: return 1
    elif c1 in vowels and c2 in vowels: return 0.5
    elif c1 not in vowels and c2 not in vowels: return 0.6
    else: return 1.2

if len(sys.argv) < 3:
    print('usage error, requires two strings to find distance on',file=sys.stderr)
    sys.exit(1)

(cost,(ext1,ext2)) = needleman_wunsch(sys.argv[1],sys.argv[2],penalty_func,null_char)
print('Cost: ' + str(cost))
print(ext1)
print(ext2)
