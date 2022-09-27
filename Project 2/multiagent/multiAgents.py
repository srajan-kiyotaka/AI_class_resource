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


##############################################
##                   Lab 2                  ##
##                                          ##
##    Secondary Authors: Team               ##
##                                          ##
##    Member 1:                             ##
##       Name: Srajan chourasia             ##
##       Roll No.: 2003135                  ##
##                                          ##
##    Member 2:                             ## 
##       Name: Shivam                       ##
##       Roll No.: 2003132                  ##
##                                          ##
##    Branch: CSE                           ##
##                                          ##
##    Date: 28/09/2022                      ##
##                                          ##
##############################################


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

        ## "*** YOUR CODE HERE ***" ##
        
        # Get the status of the game, i.e. whether the pacman win the game or not.
        # if the pacman wins the game return the highest number possible, i.e.
        # + infinity.(because the higher numbers are better state value.)
        if successorGameState.isWin():
            return float("inf")

        # Get the status of the game, i.e. whether the pacman win the game or not.
        # if the pacman lose the game return the lowest number possible, i.e.
        # - infinity.(because the lower numbers are worst state value.)
        if successorGameState.isLose():
            return float("-inf")    

        # newGhostStates contains the new status information about the ghosts, 
        # after the specified pacman move. We iterate over each ghost and calculates
        # the manhattan distance between the pacman and the ghost and if the ghost 
        # is very near the pacman then return the lowest value possible i.e. -infinity. 
        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) <= 2:
                return float("-inf")

        # a variable to store the minimum manhattan distance between the food pallet 
        # and the pacman. To get the manhattan distance of the nearest food pallet.
        nearestFoodDistance = 1000

        # iterate over each food palets after the specified pacman move(new state of 
        # the food palets), and store the minimum value between the current minimum 
        # manhattan value and the manhattan distance between the food pallet and the
        # new pacman position. To get the manhattan distance of the nearest food pallet.
        for food in list(newFood.asList()):
            nearestFoodDistance = min(nearestFoodDistance, util.manhattanDistance(food, newPos))

        # a variable to store the points of evaluation of the food status after the
        # specified pacman move(new state of the food palets).
        foodPalletStatus = 0

        # evaluation method: `if the total number of food pallets get reduces from the 
        # current state to the next state after the specified pacman move, then make the
        # foodPalletStatus value to be + 500, but if there is no change then make the 
        # foodPalletStatus equal to -150.`
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            foodPalletStatus = 500
        else:
            foodPalletStatus = -150
        
        ## These are some of the Evaluation Functions We tried ## 
        # return successorGameState.getScore() - 5 * nearestFoodDistance + foodPalletStatus
        # return successorGameState.getScore() + foodPalletStatus - 4 * nearestFoodDistance
        # return successorGameState.getScore() - foodPalletStatus - 4 * nearestFoodDistance
        # return successorGameState.getScore() + foodPalletStatus - nearestFoodDistance
        # return successorGameState.getScore() + foodPalletStatus - (1/4) * nearestFoodDistance
        # return successorGameState.getScore() + foodPalletStatus - 4 * nearestFoodDistance
        ## Finally We get the reasonable good performance using this Evaluation Function! ##
        return successorGameState.getScore() + foodPalletStatus - 2.5 * nearestFoodDistance


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
            return self.getMaxStateValue(gameState, currentDepth)

        # If this state action is to be taken from the 'Min Agent'. 
        # Then calculate and return the Min State value. 
        else:
            return self.getMinStateValue(gameState, currentDepth, agentIndex)


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
                self.getStateValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0, alpha, beta))
            else:
                minStateValue = min(minStateValue, 
                self.getStateValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex + 1, alpha, beta))
            
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
            return self.getAvgSumStateValue(gameState,currentDepth,agentIndex)


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


    def getAvgSumStateValue(self, gameState, currentDepth, agentIndex):
        """
        Calculate and return the Average State Value.
        """

        # Here we will not calculate the minimum value of the state instead,
        # we will calculate the average value of the state and store that value
        # in the avgStateValue variable.
        avgSumStateValue = 0

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
                avgSumStateValue = avgSumStateValue + self.getStateValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0)
            else:
                avgSumStateValue = avgSumStateValue + self.getStateValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex + 1)
        
        return avgSumStateValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    ## "*** YOUR CODE HERE ***" ##

    # Get the status of the game, i.e. whether the pacman win the game or not.
    # if the pacman wins the game return the highest number possible, i.e.
    # + infinity.(because the higher numbers are better state value.)
    if currentGameState.isWin():
        return float("inf")

    # Get the status of the game, i.e. whether the pacman win the game or not.
    # if the pacman lose the game return the lowest number possible, i.e.
    # - infinity.(because the lower numbers are worst state value.)
    if currentGameState.isLose():
        return float("-inf")  

    # Get the evaluation score for the current game state. `scoreEvaluationFunction`
    # is a default evaluation function which just returns the score of the state. 
    # The evaluation score is the same one displayed in the Pacman GUI.
    evaluationScore = scoreEvaluationFunction(currentGameState)

    # Get the Position/Coordinates of all the food pallets present in the current
    # game state and store it in the foodPalletsPosition variable.
    foodPalletsPosition = currentGameState.getFood()
    
    # Get the Position/Coordinate of pacman in the current game state and store
    # the coordinate in the pacmanPosition variable. 
    pacmanPosition = currentGameState.getPacmanPosition()

    # Get the total count of food pallets present in the current game state.
    totalFoodPalletLeft = currentGameState.getNumFood()

    # To store the nearest and farthest ghost manhattan distance from the 
    # pacman in nearestGhostDistance and farthestGhostDistance. Initialise
    # nearestGhostDistance = 1000 and farthestGhostDistance = -1000. 
    nearestGhostDistance = 1000
    farthestGhostDistance = -1000
    
    # Iterate over all the ghosts(Min Agents) and calculate and update the
    # nearestGhostDistance and farthestGhostDistance with the minimum and the
    # maximum of the current nearestGhostDistance value and the new manhattan
    # distance between pacman and the ghost.
    for i in range(1, currentGameState.getNumAgents()):
        
        nearestGhostDistance = min(nearestGhostDistance,
            util.manhattanDistance(currentGameState.getGhostPosition(i), pacmanPosition))
        
        farthestGhostDistance = max(farthestGhostDistance,
            util.manhattanDistance(currentGameState.getGhostPosition(i), pacmanPosition))
        
        # If the nearestGhostDistance is less then equal to 2 then directly return
        # the lowest possible value, i.e. - Infinity.
        if nearestGhostDistance <= 2:
            return float("-inf")

    # To store the nearest and farthest food pallet's manhattan distance from
    # the pacman in nearestFoodDistance and farthestFoodDistance. Initialise
    # nearestFoodDistance = 1000 and farthestFoodDistance = -1000. 
    nearestFoodDistance = 1000
    farthestFoodDistance = -1000

    # Iterate over all the food pallets and calculate and update the
    # nearestFoodDistance and farthestFoodDistance with the minimum and the 
    # maximum of the current nearestFoodDistance value and the new manhattan
    # distance between pacman and the Food.
    for food in list(foodPalletsPosition.asList()):
        nearestFoodDistance = min(nearestFoodDistance, util.manhattanDistance(food, pacmanPosition))
        farthestFoodDistance = max(farthestFoodDistance, util.manhattanDistance(food, pacmanPosition))

    ## These are some of the Evaluation Functions We tried ## 
    # return evaluationScore - 2.4 * nearestFoodDistance + nearestGhostDistance + farthestGhostDistance - farthestFoodDistance - 3 * totalFoodPalletLeft
    # return evaluationScore - 1.5 * nearestFoodDistance + 2 * nearestGhostDistance + farthestGhostDistance - 0.8 * farthestFoodDistance - 6 * totalFoodPalletLeft
    # return evaluationScore - 2.1 * nearestFoodDistance + nearestGhostDistance + farthestGhostDistance - farthestFoodDistance - 4 * totalFoodPalletLeft
    # return evaluationScore - 2 * nearestFoodDistance + nearestGhostDistance + 0.75 * farthestGhostDistance - farthestFoodDistance - 7.5 * totalFoodPalletLeft
    # return evaluationScore - 1.8 * nearestFoodDistance + nearestGhostDistance + farthestGhostDistance - 0.9 * farthestFoodDistance - 6 * totalFoodPalletLeft
    # return evaluationScore - 2 * nearestFoodDistance + 1.1 * nearestGhostDistance + 0.8 * farthestGhostDistance - farthestFoodDistance - 7 * totalFoodPalletLeft
    ## Finally We get the reasonable good performance using this Evaluation Function! ##
    return evaluationScore - 2 * nearestFoodDistance + nearestGhostDistance + farthestGhostDistance - farthestFoodDistance - 7.8 * totalFoodPalletLeft

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
