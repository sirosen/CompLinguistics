#!/usr/bin/python
def needleman_wunsch(s1,s2,penalty_func,null_char):
    def get_alignment(matrix):
        l1 = list(s1)
        l2 = list(s2)
        s1_mapping = ''
        s2_mapping = ''

        (i,j) = (len(s1),len(s2))
        while i >= 0 or j >= 0:
            if i < 0 or \
                    matrix[(i,j)] == (matrix[(i-1,j)] + penalty_func(l2[j],null_char)):
                s1_mapping = '-' + s1_mapping
                s2_mapping = l2[j] + s2_mapping
                j -= 1
            elif j < 0 or \
                    matrix[(i,j)] == (matrix[(i,j-1)] + penalty_func(l1[i],null_char)):
                s1_mapping = l1[i] + s1_mapping
                s2_mapping = '-' + s2_mapping
                i -= 1
            else: #i > 0 and j > 0 and no insertion or deletion
                s1_mapping = l1[i] + s1_mapping
                s2_mapping = l2[j] + s2_mapping

    matrix = {}
    for i,x in enumerate(s1):
        matrix[(i,0)] = penalty_func(x,null_char)
    for j,x in enumerate(s2):
        matrix[(0,j)] = penalty_func(x,null_char)
    for i,c1 in enumerate(s1):
        if i is 0: continue
        for j,c2 in enumerate(s2):
            if j is 0: continue
            match = matrix[(i-1,j-1)] + penalty_func(c1,c2)
            delete = matrix[(i-1,j)] + penalty_func(c1,null_char)
            insert = matrix[(i,j-1)] + penalty_func(c2,null_char)
            matrix[(i,j)] = min(match, delete, insert)

    return (matrix[(len(s1),len(s2))], get_alignment(matrix))
