#!/usr/bin/env python
#Author: Stephen Rosen


from __future__ import division

#Takes in a value and a mapping from keys onto probabilities, and returns the key selected by that value
def select_from_probability_dict(value, probability_dict):
    running_total = 0
    #sort the dictionary by keys, so that we are guaranteed consistent behavior
    #even when values are added & removed from the dict
    for k in sorted(probability_dict):
        v = probability_dict[k]
        running_total += v
        if running_total >= value: return k
    return None
        

class HMM(object):
    #Create a new HMM
    def __init__(self,numstates,alphabet,pi_values={},transition_map={},emission_map={}):
        """
        Creates a new Hidden Markov Model.

        Args:
            numstates
            The number of states in the HMM

            alphabet
            A set of characters

        KWArgs:
            pi_values
            A dictionary mapping states (integers from 0 to the number of states) to starting probabilities.
            Any unassigned probabilities will be taken to be equal portions of the remaining probability mass.

            transition_map
            A map from ordered index pairs onto transition probabilities.
            Any unassigned probabilities will be taken to be equal portions of the remaining probability mass.

            emission_map
            A map from states to emission probabilities for each symbol.
            Emission probabilities are themselves represented by a map from symbols to probabilities.
            Any unassigned probabilities will be taken to be equal portions of the remaining probability mass.
        """
        self.states = frozenset(range(numstates))
        self.alphabet = frozenset(alphabet)

        assert (len(self.alphabet) != 0)

        #Initialize the pi values
        #start the probability mass at 1, and reduce it for every element of pi_values
        mass = 1
        numvalues = numstates
        self.pi_values = {}
        for (k,v) in pi_values.items():
            self.pi_values[k] = v
            mass -= v
            numvalues -= 1
        #assign the remaining mass evenly
        if numvalues > 0:
            p = mass/numvalues
            for s in self.states.difference(pi_values.keys()):
                self.pi_values[s] = p

        #Initialize the transition matrix
        self.transition_map = {}
        for ((s1,s2),v) in transition_map.items():
            self.transition_map[(s1,s2)] = v
        #As with pi_values, we compute the reserve probability mass, but we must do so on a state by state basis
        for s1 in self.states:
            mass = 1
            numvalues = numstates
            for s2 in self.states:
                if (s1,s2) in self.transition_map:
                    mass -= self.transition_map[(s1,s2)]
                    numvalues -= 1
            #and assign that remaining mass
            if numvalues > 0:
                p = mass / numvalues
                for s2 in self.states:
                    if (s1,s2) not in self.transition_map:
                        self.transition_map[(s1,s2)] = p

        #Initialize the emission map
        self.emission_map = {}
        for s in self.states:
            #If the state has nothing specified, it takes on the reasonable default
            #assign equal probability to each letter in each state
            if s not in emission_map:
                p = 1/len(self.alphabet)
                self.emission_map[x] = { l:p for l in self.alphabet }
            else:
                mass = 1
                numvalues = len(self.alphabet)
                state_map = emission_map[s]
                self.emission_map[s] = {}
                #Write all of the values that we have into the map
                for l in state_map:
                    v = state_map[l]
                    self.emission_map[s][l] = v
                    mass -= v
                    numvalues -= 1
                #Assign the remainder probability
                if numvalues > 0:
                    p = mass / numvalues
                    for l in self.alphabet.difference(state_map.keys()):
                        self.emission_map[s][l] = p
