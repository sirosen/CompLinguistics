#!/usr/bin/python

#Beware: Large Memory Footprint Potential!

from __future__ import print_function, division
import argparse, sys, itertools, collections

def overlap(s1,s2):
    def letter_pairs(s):
        res=set()
        prev=None
        for x in enumerate(s):
            if not prev:
                prev = x
            else:
                res.add((prev,x))
                prev = x
        return res
    return len(letter_pairs(s1).union(letter_pairs(s2)))

def set_overlap(ss):
    return sum(overlap(a,b) for (a,b) in itertools.combinations(ss,2))

def surprise_value(ss):
    from dist import simple_dist
    from math import factorial as f
    if len(ss) < 2: return 0

    l = len(ss)
    nC2 = f(l) / 2 / f(l-2)
    total_dist = sum(simple_dist(a,b) for (a,b) in itertools.combinations(ss,2))
    return total_dist / nC2

parser = argparse.ArgumentParser()
parser.add_argument('-f','--filename',type=argparse.FileType('r'),help='Dictionary file.')
args = parser.parse_args(sys.argv[1:])
filename = args.filename

words = set()
for line in filename:
    words.add(line.strip())

filename.close()

counts_to_perms = {}
for word in words:
    letter_count = collections.Counter(word)
    fset = frozenset(letter_count.items())
    if fset in counts_to_perms:
        counts_to_perms[fset].add(word)
    else:
        counts_to_perms[fset] = set()

def p_len(p):
    if len(p) is 0: return 0
    else: return len(list(p)[0]) #dirty, works

#use cmp instead of simple tuple key to prevent computation
#of set overlap in unnecessary cases
def perm_cmp(p1,p2):
    n1 = len(p1)
    n2 = len(p2)
    if n1 == n2:
        l1 = p_len(p1)
        l2 = p_len(p2)
        if l1 == l2:
            s1 = set_overlap(p1)
            s2 = set_overlap(p2)
            return cmp(s1,s2)
        else: return cmp(l1,l2)
    else: return cmp(n1,n2)

for perm in sorted((p for p in counts_to_perms.values() if p_len(p) >= 6),cmp=perm_cmp):
    print(perm)

print('--------------------')
print('Most Surprising 100!')
print('--------------------')
for perm in sorted(counts_to_perms.values(),key=surprise_value,reverse=True)[:100]:
    print(perm)
