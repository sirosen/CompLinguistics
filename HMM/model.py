#!/usr/bin/env python
#Author: Stephen Rosen


from __future__ import print_function, division
from random import random

from utils import memoize, normalize

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
As noted in X. Li, et al, in "Training Hidden Markov Models with
Multiple Observations -- A Combinatorial Method", even though we
have built our HMM with respect to a single observation, we can
operate over sums of probabilities for multiple observations.
"""
class HMM(object):
    #Create a new HMM
    def __init__(self,numstates,alphabet,fixed=True,pi_values={},transition_map={},emission_map={}):
        """
        Creates a new Hidden Markov Model.

        Args:
            numstates
            The number of states in the HMM

            alphabet
            A set of characters

        KWArgs:
            fixed
            Sets the probability distribution to be fixed or random.

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
        if numvalues > 0:
            #assign the remaining mass evenly
            if fixed:
                p = mass/numvalues
                for s in self.states.difference(pi_values.keys()):
                    self.pi_values[s] = p
            #If the probability distribution is not fixed, distribute the remaining mass randomly
            else:
                d = {}
                for s in self.states.difference(pi_values.keys()):
                    d[s] = random()
                normalize(d)
                for s in d:
                    self.pi_values[s] = mass*d[s]

        #Initialize the transition matrix
        self.transition_map = {}
        for (s1,d) in transition_map.items():
            self.transition_map[s1] = {}
            for s2 in d:
                self.transition_map[s1][s2] = d[s2]
        #As with pi_values, we compute the reserve probability mass, but we must do so on a state by state basis
        for s1 in self.states:
            if s1 not in transition_map:
                self.transition_map[s1] = {}
            mass = 1
            numvalues = numstates
            for s2 in self.states:
                if s2 in self.transition_map[s1]:
                    mass -= self.transition_map[s1][s2]
                    numvalues -= 1
            if numvalues > 0:
                #and assign that remaining mass evenly
                if fixed:
                    p = mass / numvalues
                    for s2 in self.states:
                        if s2 not in self.transition_map[s1]:
                            self.transition_map[s1][s2] = p
                #If the probability distribution is not fixed, distribute the remaining mass randomly
                else:
                    d = {}
                    for s2 in self.states:
                        if s2 not in self.transition_map[s1]:
                            d[s2] = random()
                    normalize(d)
                    for s2 in d:
                        self.transition_map[s1][s2] = mass*d[s2]

        #Initialize the emission map
        self.emission_map = {}
        for s in self.states:
            #If the state has nothing specified, it takes on the reasonable default
            if s not in emission_map:
                #assign equal probability to each letter in each state
                if fixed:
                    p = 1/len(self.alphabet)
                    self.emission_map[s] = { l:p for l in self.alphabet }
                #If the probability distribution is not fixed, distribute the remaining mass randomly
                else:
                    d = { k:random() for k in self.alphabet }
                    normalize(d)
                    self.emission_map[s] = {}
                    for k in d:
                        self.emission_map[s][k] = mass*d[k]

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
                    if fixed:
                        p = mass / numvalues
                        for l in self.alphabet.difference(state_map.keys()):
                            self.emission_map[s][l] = p
                    #If the probability distribution is not fixed, distribute the remaining mass randomly
                    else:
                        d = { k:random() for k in self.alphabet.difference(state_map.keys()) }
                        normalize(d)
                        for k in d:
                            self.emission_map[s][k] = mass*d[k]

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
        def alpha_helper(i,t,O):
            #assert that the world isn't broken
            assert (t >= 0)
            assert (t <= len(O))
            #grab the base case
            if t == 0:
                return self.pi_values[i]
            #recursive application, equation 9.10 of Manning and Schutze
            else:
                return sum(alpha_helper(j,t-1,O)*trans[j][i]*em[j][O[t-1]] for j in states)

        return alpha_helper(state,time,O)

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
        def beta_helper(i,t,O):
            #print('State: ' + str(i))
            #print('Time: ' + str(t))
            #assert that the world is safe
            assert (t >= 0)
            assert (t <= len(O))
            #grab the base case
            if t == len(O):
                return 1
            #recursive application, equation 9.11
            else:
                if O[t] == ',':
                    print("HERE")
                    import sys
                    sys.exit(1)
                return sum(beta_helper(j,t+1,O)*trans[i][j]*em[i][O[t]] for j in states)

        return beta_helper(state,time,O)

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
    We assume the independence of each word in the corpus.
    By so doing, we allow the use of the Levinson training equations below.
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
        num = sum(sum(self.p(i, j, t, O) for t in xrange(len(O)-1)) for O in corpus)
        denom = sum(sum(self.gamma(i, t, O) for t in xrange(len(O)-1)) for O in corpus)

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

    def iterate_until_convergence(self,delta,corpus,max_iterations=100):
        """
        Run maximization on the model until all parameters converge within delta.
        Returns True if we converged, False otherwise

        Args:
            delta
            The convergence parameter

            corpus
            An iterable collection of strings

        KWArgs:
            max_iterations
            The maximum number of times that the model will run before giving up on convergence
        """
        states = self.states

        #Tests if all parameters fall within delta
        def delta_condition(new_pi, new_trans, new_em):
            for i in new_pi:
                if abs(new_pi[i]-self.pi_values[i]) > delta: return False
            for i in new_trans:
                for j in new_trans[i]:
                    if abs(new_trans[i][j]-self.transition_map[i][j]) > delta: return False
            for i in new_em:
                for k in new_em[i]:
                    if abs(new_em[i][k]-self.emission_map[i][k]) > delta: return False
            return True

        #Run at most max_iterations steps
        for i in xrange(max_iterations):
            #recalculate and normalize each set of values
            pi_vals = { i:self.recalculate_pi(i,corpus) for i in states }
            trans_vals = { i:{j: self.recalculate_transition(i,j,corpus) for j in states } for i in states }
            em_vals = { i:{k: self.recalculate_emission(i,k,corpus) for k in self.alphabet } for i in states }
            normalize(pi_vals)
            for i in states:
                normalize(trans_vals[i])
                normalize(em_vals[i])

            #if we have converged return true
            if delta_condition(pi_vals,trans_vals,em_vals):
                return True
            #otherwise, switch to the new model
            else:
                self.pi_values = pi_vals
                self.transition_map = trans_vals
                self.emission_map = em_vals

        #if we reach this point, we have not converged, so return False
        return False

    def dump_state(self):
        print('Pi Values')
        print('\tState:Value')
        for i in self.pi_values:
            print('\t'+str(i)+':'+str(self.pi_values[i]))
        print('Transitions')
        print('\tState->State:Value')
        for i in self.transition_map:
            for j in self.transition_map[i]:
                print('\t'+str(i)+'->'+str(j)+':'+str(self.transition_map[i][j]))
        print('Emissions')
        print('\tState->Symbol:Value')
        for i in self.emission_map:
            for k in self.emission_map[i]:
                print('\t'+str(i)+'->'+str(k)+':'+str(self.emission_map[i][k]))


    def soft_counts(self, letter, corpus):
        #state to soft count
        state_to_count = {}
        for i in self.states:
            tmp = 0
            for word in words:
                for t in xrange(len(word)):
                    if word[t] == letter:
                        for j in self.states:
                            tmp += self.p(i,j,t,word)
            state_to_count[s] = tmp
        return state_to_count

def corpus_from_file(alphabet,fname):
    from string import punctuation, digits
    f = open(fname,'r')
    words = f.read().lower().strip().translate(None,punctuation+digits).split()
    corpus = set('#'+w+'#' for w in words)
    f.close()
    return corpus

if __name__ == '__main__':
    import sys
    from string import lowercase
    if len(sys.argv) < 2:
        print('USAGE: model.py [filename]',file=sys.stderr)
        sys.exit(2)

    alphabet = lowercase+'#'
    corpus = corpus_from_file(alphabet,sys.argv[1])

    h = HMM(2,alphabet,fixed=False)
    h.iterate_until_convergence(0.05,corpus)
    h.dump_state()
