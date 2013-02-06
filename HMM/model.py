#!/usr/bin/env python
#Author: Stephen Rosen


from __future__ import division
from random import random

from utils import memoize


def select_from_probability_dict(value, probability_dict):
    """
    Takes in a value and a mapping from keys onto probabilities, and returns the key selected by that value

    Args:
        value
        A probability in (0,1)

        probability_dict
        A dictionary mapping values onto their probabilities
    """
    running_total = 0
    #sort the dictionary by keys, so that we are guaranteed consistent behavior
    #even when values are added & removed from the dict
    for k in sorted(probability_dict):
        v = probability_dict[k]
        running_total += v
        if running_total >= value: return k
    return None


"""
As noted in X. Li, et al, in "Training Hidden Markov Models with Multiple Observations -- A Combinatorial Method",
even though we have built our HMM with respect to a single observation, we can operate over the sums of probabilities for multiple observations
"""
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
            A map from state pairs onto transition probabilities.
            Transition probabilities are themselves represented by a map from states to probabilities.
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
        for (s1,d) in transition_map.items():
            self.transition_map[s1] = {}
            for s2 in d:
                self.transition_map[s1][s2] = d[s2]
        #As with pi_values, we compute the reserve probability mass, but we must do so on a state by state basis
        for s1 in self.states:
            mass = 1
            numvalues = numstates
            for s2 in self.states:
                if s1 in self.transition_map and s2 in self.transition_map[s1]:
                    mass -= self.transition_map[s1][s2]
                    numvalues -= 1
            #and assign that remaining mass
            if numvalues > 0:
                p = mass / numvalues
                for s2 in self.states:
                    if s1 not in self.transition_map or s2 not in self.transition_map[s1]:
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

        self.current_state = select_from_probability_dict(random(),self.pi_values)

    def change_state(self):
        """
        Performs a state transition
        """
        transitions = self.transition_map[self.current_state]
        self.current_state = select_from_probability_dict(random(),transitions)

    def emit_symbol(self):
        """
        Performs a symbol emission
        """
        emissions = self.emission_map[self.current_state]
        return select_from_probability_dict(random(),emissions)

    def step(self):
        """
        Performs a single action for the HMM, 
        """
        self.emit_symbol()
        self.change_state()

    def alpha(self,state,time,observation):
        """
        Calculates the forward variable alpha_{state}(time)

        Args:
            state
            The state being measured

            time
            The timepoint at which the probability of the state is taken

            observation
            An observed string from the corpus
        """
        trans = self.transition_map
        em = self.emission_map
        states = self.states
        O = observation

        @memoize
        def alpha_helper(i,t):
            #assert that the world isn't broken
            assert (t >= 0)
            assert (t <= len(O))
            #grab the base case
            if t == 0:
                return self.pi_values[i]
            #recursive application, equation 9.10 of Manning and Schutze
            else:
                return sum(alpha_helper(j,t-1)*trans[j][i]*em[j][O[t-1]] for j in states)

        return alpha_helper(state,time)

    def beta(self,state,time,observation):
        """
        Calculates the backward variable beta_{state}(time)

        Args:
            state
            The state being measured

            time
            The timepoint at which the probability of the state is taken

            observation
            An observed string from the corpus
        """
        trans = self.transition_map
        em = self.emission_map
        states = self.states
        O = observation

        @memoize
        def beta_helper(i,t):
            #assert that the world is safe
            assert (t >= 0)
            assert (t <= len(O))
            #grab the base case
            if t == len(O):
                return 1
            #recursive application, equation 9.11
            else:
                return sum(beta_helper(j,t+1)*trans[i][j]*em[i][O[t]] for j in states)

        return beta_helper(state,time)

    def p(self, i, j, time, observation):
        """
        The probability of transitioning from i to j at a given time

        Args:
            i
            The state we transition from

            j
            The state we transition to

            time
            The fixed timepoint

            observation
            An observed string from the corpus
        """
        trans = self.transition_map
        em = self.emission_map
        states = self.states
        O = observation

        num = self.alpha(i,time,O)*trans[i][j]*em[i][O[time]]*self.beta(j,time,O)
        denom = sum(self.alpha(m,time,O)*trans[m][n]*em[m][O[time]]*self.beta(n,time+1,O) for n in states for m in states)
        return num / denom

    def gamma(self, i, time, observation):
        """
        The probability of being in state i at a given time

        Args:
            i
            The state in question

            time
            The fixed timepoint

            observation
            An observed string from the corpus
        """
        states = self.states
        O = observation

        #Equation 9.13 from Manning and Schutze
        num = self.alpha(i,time,O) * self.beta(i,time,O)
        denom = sum(self.alpha(j,time,O) * self.beta(j,time,O) for j in states)

        return num / denom


"""
We assume the independence of each word in the corpus, and by so doing allow the use of the Levinson training equations below.
"""


    def recalculate_pi(self, i, corpus):
        """
        A reestimation of a single pi value

        Args:
            i
            The state being considered

            corpus
            An iterable collection of observations (strings)
        """
        return sum(self.gamma(i,0,O) for O in corpus) / len(corpus)

    def recalculate_transition(self, i, j, corpus):
        """
        A reestimation of a single transition probability

        Args:
            i
            The from state

            j
            The to state

            corpus
            An iterable collection of observations (strings)
        """
        num = sum(sum(self.p(i, j, t, O) for t in xrange(len(O))) for O in corpus)
        denom = sum(sum(self.gamma(i, t, O) for t in xrange(len(O))) for O in corpus)

        return num / denom

    def recalculate_emission(self, i, k, corpus):
        """
        A reestimation of a single emission probability

        Args:
            i
            The emmitting state

            k
            The emmitted symbol

            corpus
            An iterable collection of observations (strings)
        """
        num = sum(sum(self.gamma(i, t, O) for t in xrange(len(O)) if O[t] == k) for O in corpus)
        denom = sum(sum(self.gamma(i,t, O) for t in xrange(len(O))) for O in corpus)

        return num / denom
