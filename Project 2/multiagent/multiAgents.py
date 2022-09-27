# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get the status of the game, i.e. whether the pacman win the game or not
        # if the pacman wins the game return the highest number possible, i.e.
        # + infinity.(because the higher numbers are better.)
        if successorGameState.isWin():
            return float("inf")

        # newGhostStates contains the new status information about the ghosts, 
        # after the specified pacman move. We iterate over each ghost and calculates
        # the manhattan distance between the pacman and the ghost and if the ghost 
        # is very near the pacman then return the lowest value possible i.e. -infinity. 
        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 3:
                return float("-inf")

        # a list to store the manhattan distance between the food pallet and the pacman.
        foodDistanceList = []

        # iterate over each food palets after the specified pacman move(new state of the
        # food palets), and store the manhattan distance between the food pallet and the
        # new pacman position. 
        for food in list(newFood.asList()):
            foodDistanceList.append(util.manhattanDistance(food, newPos))

        # a variable to store the point of evaluation of the food status after the
        # specified pacman move(new state of the food palets).
        foodSuccessor = 0

        # evaluation method: `if the total number of food pallets get reduces from the 
        # current state to the next state after the specified pacman move, then make the
        # foodSuccessor value to be + 400, but if there is no change then make the 
        # foodSuccessor equal to -25.`
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            foodSuccessor = 400
        else:
            foodSuccessor = -25

        ## These are some of the Evaluation Functions We tried ## 
        # return successorGameState.getScore() - 5 * min(foodDistanceList) + foodSuccessor
        # return successorGameState.getScore() + foodSuccessor - 4 * min(foodDistanceList)
        # return successorGameState.getScore() - foodSuccessor - 4 * min(foodDistanceList)
        # return successorGameState.getScore() + foodSuccessor - max(foodDistanceList)
        # return successorGameState.getScore() + foodSuccessor - (1/4) * max(foodDistanceList)
        # return successorGameState.getScore() + foodSuccessor - 4 * min(foodDistanceList)
        ## Finally We get the reasonable good performance using this Evaluation Function! ##
        return successorGameState.getScore() + foodSuccessor - 5 * min(foodDistanceList)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        ## "*** YOUR CODE HERE ***" ##

        # set the max state value to be the lowest possible value, i.e. 
        # -infinity initially. We will store the maximum state value.
        maxStateValue = float("-inf")

        # We will store the action for the max state.
        # we initially make the action as Direction.STOP which means stop.
        maxStateActions = Directions.STOP

        # iterate over all possible action and get the state value and actions
        # to reach that state for each possible action from the current state.  
        for action in gameState.getLegalActions(0):
            
            # get the new state after taking action Ai.
            nextState = gameState.generateSuccessor(0, action)
            
            # We will use the get value function to calculate the next state value.
            nextValue = self.getStateValue(nextState, 0, 1)

            # if this states value is greater then the current max state value then
            # update the maxStateValue and maxStateActions. 
            if nextValue > maxStateValue:
                maxStateValue = nextValue
                maxStateActions = action
        
        return maxStateActions

        util.raiseNotDefined()

    def getStateValue(self, gameState, currentDepth, agentIndex):
        """
        Returns the state value.
        """
        # Check for the terminal state. if this is the terminal state then
        # return the evaluation value of that state.
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # If this state action is to be taken from the 'Max Agent'.
        # Then calculate and return the Max State value. 
        elif agentIndex == 0:
            return self.getMaxStateValue(gameState,currentDepth)

        # If this state action is to be taken from the 'Min Agent'. 
        # Then calculate and return the Min State value. 
        else:
            return self.getMinStateValue(gameState,currentDepth,agentIndex)

    def getMaxStateValue(self, gameState, currentDepth):
        """
        Calculate and return the Max State Value.
        """

        # set the maximum state value as -infinity initially.
        # Which will store the maximum state value.
        maxStateValue = float("-inf")

        # iterate over all possible action and get the maximum state value to
        # reach that state for each possible action from the current state.
        for action in gameState.getLegalActions(0):
            
            # store the maximum state value and pass the next state value with
            # making the next state agent as 'Min Agent'. 
            maxStateValue = max(maxStateValue, self.getStateValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        
        return maxStateValue

    def getMinStateValue(self, gameState, currentDepth, agentIndex):
        """
        Calculate and return the Min State Value.
        """

        # set the minimum state value as +infinity initially.
        # Which will store the minimum state value.
        minStateValue = float("inf")

        # iterate over all possible action and get the minimum state value to
        # reach that state for each possible action from the current state.  
        for action in gameState.getLegalActions(agentIndex):
            
            # first go through all the 'Min Agent' by keeping the depth of the 
            # tree constant, then if we have gone through all the 'Min Agents' 
            # state then go to the next level of the state for the 'Max Agent' 
            # and increase the depth of the tree.  
            if agentIndex == gameState.getNumAgents() - 1:
                minStateValue = min(minStateValue, 
                self.getStateValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0))
            else:
                minStateValue = min(minStateValue, 
                self.getStateValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex + 1))
        
        return minStateValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        and using alpha-beta pruning. 
        """
        ## "*** YOUR CODE HERE ***" ##

        # set the max state value to be the lowest possible value, i.e. 
        # -infinity initially. We will store the maximum state value.
        maxStateValue = float("-inf")

        # alpha is the Maximum best option on the path to the root.
        # initially set it to the lowest possible value i.e. - infinity.
        alpha = float("-inf")
        
        # alpha is the Minimum best option on the path to the root.
        # initially set it to the highest possible value i.e. + infinity.
        beta = float("inf")
        
        # We will store the action for the max state.
        # we initially make the action as Direction.STOP which means stop.
        maxStateActions = Directions.STOP
        
        # iterate over all possible action and get the state value and actions
        # to reach that state for each possible action from the current state.
        for action in gameState.getLegalActions(0):

            # get the new state after taking action Ai.
            nextState = gameState.generateSuccessor(0, action)
            
            # We will use the get value function to calculate the next state value.
            nextValue = self.getStateValue(nextState, 0, 1, alpha, beta)

            # if this states value is greater then the current max state value then
            # update the maxStateValue and maxStateActions.
            if nextValue > maxStateValue:
                maxStateValue = nextValue
                maxStateActions = action
            
            # update the alpha value with the max of alpha and maxStateValue.
            alpha = max(alpha, maxStateValue)
        
        return maxStateActions

        util.raiseNotDefined()


    def getStateValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        """
        Returns the state value using alpha-beta pruning.
        """

        # Check for the terminal state. if this is the terminal state then
        # return the evaluation value of that state.
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # If this state action is to be taken from the 'Max Agent'.
        # Then calculate and return the Max State value.
        elif agentIndex == 0:
            return self.getMaxStateValue(gameState,currentDepth,alpha,beta)

        # If this state action is to be taken from the 'Min Agent'. 
        # Then calculate and return the Min State value. 
        else:
            return self.getMinStateValue(gameState,currentDepth,agentIndex,alpha,beta)


    def getMaxStateValue(self, gameState, currentDepth, alpha, beta):
        """
        Calculate and return the Max State Value using alpha-beta pruning.
        """
        # set the maximum state value as -infinity initially.
        # Which will store the maximum state value.
        maxStateValue = float("-inf")

        # iterate over all possible action and get the maximum state value to
        # reach that state for each possible action from the current state.
        for action in gameState.getLegalActions(0):

            # store the maximum state value and pass the next state value with
            # making the next state agent as 'Min Agent'. 
            maxStateValue = max(maxStateValue,
             self.getStateValue(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta))
            
            # Check if the max state value is more then beta then no need to
            # check further actions and return the max state value, in this way 
            # we skip other action state whose calculation is not necessary.
            # This is known as alpha beta pruning.
            if maxStateValue > beta:
                return maxStateValue

            # update the alpha value with the max of alpha and maxStateValue.
            alpha = max(alpha, maxStateValue)

        return maxStateValue


    def getMinStateValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        """
        Calculate and return the Min State Value using alpha-beta pruning.
        """

        # set the minimum state value as +infinity initially.
        # Which will store the minimum state value.
        minStateValue = float("inf")

        # iterate over all possible action and get the minimum state value to
        # reach that state for each possible action from the current state.
        for action in gameState.getLegalActions(agentIndex):

            # first go through all the 'Min Agent' by keeping the depth of the 
            # tree constant, then if we have gone through all the 'Min Agents' 
            # state then go to the next level of the state for the 'Max Agent' 
            # and increase the depth of the tree.  
            if agentIndex == gameState.getNumAgents() - 1:
                minStateValue = min(minStateValue, 
                self.getStateValue(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0, alpha, beta))
            else:
                minStateValue = min(minStateValue, 
                self.getStateValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1, alpha, beta))
            
            # Check if the min state value is less then alpha then no need to
            # check further actions and return the min state value, in this way 
            # we skip other action state whose calculation is not necessary.
            # This is known as alpha beta pruning.
            if minStateValue < alpha:
                return minStateValue

            # update the beta value with the min of beta and minStateValue.
            beta = min(beta, minStateValue)
        
        return minStateValue


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        ## "*** YOUR CODE HERE ***" ##

        # set the max state value to be the lowest possible value, i.e. 
        # -infinity initially. We will store the maximum state value.
        maxStateValue = float("-inf")

        # We will store the action for the max state.
        # we initially make the action as Direction.STOP which means stop.
        maxStateActions = Directions.STOP

        # iterate over all possible action and get the state value and actions
        # to reach that state for each possible action from the current state.  
        for action in gameState.getLegalActions(0):
            
            # get the new state after taking action Ai.
            nextState = gameState.generateSuccessor(0, action)
            
            # We will use the get value function to calculate the next state value.
            nextValue = self.getStateValue(nextState, 0, 1)

            # if this states value is greater then the current max state value then
            # update the maxStateValue and maxStateActions. 
            if nextValue > maxStateValue:
                maxStateValue = nextValue
                maxStateActions = action
        
        return maxStateActions

        util.raiseNotDefined()

    def getStateValue(self, gameState, currentDepth, agentIndex):
        """
        Returns the state value.
        """
        # Check for the terminal state. if this is the terminal state then
        # return the evaluation value of that state.
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # If this state action is to be taken from the 'Max Agent'.
        # Then calculate and return the Max State value. 
        elif agentIndex == 0:
            return self.getMaxStateValue(gameState,currentDepth)

        # If this state action is to be taken from the 'Min Agent'. 
        # Then calculate and return the Min State value. 
        else:
            return self.getAvgStateValue(gameState,currentDepth,agentIndex)


    def getMaxStateValue(self, gameState, currentDepth):
        """
        Calculate and return the Max State Value.
        """

        # set the maximum state value as -infinity initially.
        # Which will store the maximum state value.
        maxStateValue = float("-inf")

        # iterate over all possible action and get the maximum state value to
        # reach that state for each possible action from the current state.
        for action in gameState.getLegalActions(0):
            
            # store the maximum state value and pass the next state value with
            # making the next state agent as 'Min Agent'. 
            maxStateValue = max(maxStateValue,
            self.getStateValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        
        return maxStateValue

    def getAvgStateValue(self, gameState, currentDepth, agentIndex):
        """
        Calculate and return the Average State Value.
        """

        # Here we will not calculate the minimum value of the state instead,
        # we will calculate the average value of the state and store that value
        # in the avgStateValue variable.
        avgStateValue = 0

        # iterate over all possible action and add all the state values to get 
        # the Average state value for each possible action from the current 
        # state. We are adding because the terminal state values from any state
        # will be either positive or negative, so when we add these values then
        # we are getting a kind of average value for that state. 
        for action in gameState.getLegalActions(agentIndex):

            # first go through all the 'Min Agent' by keeping the depth of the 
            # tree constant, then if we have gone through all the 'Min Agents' 
            # state then go to the next level of the state for the 'Max Agent' 
            # and increase the depth of the tree.  
            if agentIndex == gameState.getNumAgents() - 1:
                avgStateValue = avgStateValue + self.getStateValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0)
            else:
                avgStateValue = avgStateValue + self.getStateValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex + 1)
        
        return avgStateValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
