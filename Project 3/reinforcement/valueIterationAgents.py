# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        ## "*** YOUR CODE HERE ***" ##

        # Iterate self.iterations times and in each iteration update the
        # MDP state value using the previous state iteration value.
        for i in range(self.iterations):
            
            # Get the state values from the previous iteration.
            previousStatesValue = util.Counter()

            # Iterate over all the states in the MDP.
            for state in self.mdp.getStates():

                # Get all the possible actions values for the previous states.
                previousMdpActionQValue = util.Counter()
            
                # Iterate over all the possible actions from the state.
                for action in self.mdp.getPossibleActions(state):

                    # update the action Q value for the action.
                    previousMdpActionQValue[action] = self.computeQValueFromValues(state, action)

                # Update the state value from the maximum of the Q value for all actions. 
                previousStatesValue[state] = previousMdpActionQValue[previousMdpActionQValue.argMax()]

            # Update the state value after each iteration.
            for state in self.mdp.getStates():
                self.values[state] = previousStatesValue[state]




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        ## "*** YOUR CODE HERE ***" ##
        # Compute the Q-value
        qValue = 0

        # Iterate over all transition states from the given state.
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):

            # Compute the Q-value
            qValue = qValue + transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
        return qValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # "*** YOUR CODE HERE ***" #

        # Check if there exits action or not.
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None

        # get the action value.
        valuesForActions = util.Counter()

        # Iterate over all the action possible for the state.
        for action in self.mdp.getPossibleActions(state):
            valuesForActions[action] = self.computeQValueFromValues(state, action)

        return valuesForActions.argMax()

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def stateValueIterationStep(self, state):

        # store the max/best QValue.
        bestQValue = float('-inf')

        # Iterate over all possible action for the given state.
        for action in self.mdp.getPossibleActions(state):

            # calculate the QValue for a given state and action.
            QValue = self.computeQValueFromValues(state, action)
            bestQValue = max(bestQValue, QValue)
        return bestQValue

    def runValueIteration(self):
        # "*** YOUR CODE HERE ***" #
        
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                self.values[state] = self.stateValueIterationStep(state)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # "*** YOUR CODE HERE ***" #

        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state}
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                values = []
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    values.append(q_value)
                diff = abs(max(values) - self.values[state])
                pq.update(state, - diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            temp_state = pq.pop()
            if not self.mdp.isTerminal(temp_state):
                values = []
                for action in self.mdp.getPossibleActions(temp_state):
                    q_value = self.computeQValueFromValues(temp_state, action)
                    values.append(q_value)
                self.values[temp_state] = max(values)
            for p in predecessors[temp_state]:
                if not self.mdp.isTerminal(p):
                    values = []
                    for action in self.mdp.getPossibleActions(p):
                        q_value = self.computeQValueFromValues(p, action)
                        values.append(q_value)
                    diff = abs(max(values) - self.values[p])
                    if diff > self.theta:
                        pq.update(p, -diff)
