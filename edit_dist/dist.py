#!/usr/bin/python
def needleman_wunsch(s1,s2,penalty_func,null_char):
    def get_alignment(matrix):
        l1 = list(s1)
        l2 = list(s2)
        ext1 = ''
        ext2 = ''

        (i,j) = (len(s1)-1,len(s2)-1)
        while i > 0 or j > 0:
            if i < 0 or \
                    matrix[(i,j)] == (matrix[(i-1,j)] + penalty_func(l2[j],null_char)):
                ext1 = '-' + ext1
                ext2 = l2[j] + ext2
                j -= 1
            elif j < 0 or \
                    matrix[(i,j)] == (matrix[(i,j-1)] + penalty_func(l1[i],null_char)):
                ext1 = l1[i] + ext1
                ext2 = '-' + ext2
                i -= 1
            else: #i > 0 and j > 0 and no insertion or deletion
                ext1 = l1[i] + ext1
                ext2 = l2[j] + ext2
                i -= 1
                j -= 1

        return (ext1,ext2)

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

    return (matrix[(len(s1)-1,len(s2)-1)], get_alignment(matrix))

null_char = '#'
def penalty_func(c1,c2):
    if c1 == c2: return 0
    elif c1 == null_char or c2 == null_char: return 1
    else: return 0.5

x = (cost,(ext1,ext2)) = needleman_wunsch('dog','dog',penalty_func,null_char)
print(x)
