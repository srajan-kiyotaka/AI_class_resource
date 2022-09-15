# search.py
# ---------
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
##                   Lab 1                  ##
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
##    Date: 09/09/2022                      ##
##                                          ##
##############################################


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from tracemalloc import start
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # initialize an empty stack
    stack = util.Stack()

    # initialize an empty visited array
    visited = list()
    
    # initial_state is a tuple containing the start state coordinates and the sequence of 
    # actions taken till now to reach this state.
    # initially the list of action will be empty as it remains empty.
    initial_state = (problem.getStartState(), [])

    # push the initial state in the stack
    stack.push(initial_state)

    # iterate over all the elements present in the stack until you reach the goal state
    # or the stack become empty.
    while(not stack.isEmpty()):

        # pop the element from the stack, and do tuple unpacking to get the current 
        # state coordinates and the list of actions required to reach the current state.
        current_state, path_action = stack.pop()

        # if the current state is the goal state then return the list of actions
        if(problem.isGoalState(current_state)):
            return path_action

        # if the current goal is already visited then we will not push this state in the 
        # stack and continue.
        if(current_state in visited):
            continue
        
        # add the states as a visited state.
        visited.append(current_state)
        
        # iterate over all it's neighbors and push those states in the stack.
        for next_state, action, cost in problem.getSuccessors(current_state):

            # if the state is not already visited. 
            if(next_state not in visited):

                # add the action taken to reach the current state from the previous state
                # in the list of actions which contains the actions to reach the previous
                # state from the start state.
                next_path_action = path_action + [action]

                # Push the tuple in the stack. The tuple contains the coordinates of the 
                # next stage and the list of actions required to reach that stage.
                stack.push(tuple([next_state, next_path_action]))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    # initialize an empty queue
    queue = util.Queue()

    # initialize an empty visited array
    visited = list()
    
    # initial_state is a tuple containing the start state coordinates and the sequence of 
    # actions taken till now to reach this state.
    # initially the list of action will be empty as it remains empty.
    initial_state = (problem.getStartState(), [])

    # push the initial state in the queue
    queue.push(initial_state)

    # iterate over all the elements present in the queue until you reach the goal state
    # or the queue become empty.
    while(not queue.isEmpty()):
        
        # pop the element from the queue, and do tuple unpacking to get the current 
        # state coordinates and the list of actions required to reach the current state.
        current_state, path_action = queue.pop()

        # if the current state is the goal state then return the list of actions.
        if(problem.isGoalState(current_state)):
            return path_action

        # if the current goal is already visited then we will not push this state in the 
        # queue and continue.
        if(current_state in visited):
            continue
        
        # add the states as a visited state.
        visited.append(current_state)
        
        # iterate over all it's neighbors and push those states in the queue.
        for next_state, action, cost in problem.getSuccessors(current_state):
            
            # if the state is not already visited. 
            if(next_state not in visited):
            
                # add the action taken to reach the current state from the previous state
                # in the list of actions which contains the actions to reach the previous
                # state from the start state.
                next_path_action = path_action + [action]

                # Push the tuple in the stack. The tuple contains the coordinates of the 
                # next stage and the list of actions required to reach that stage.
                queue.push(tuple([next_state, next_path_action]))
        
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    # initialize an empty priority queue
    priority_queue = util.PriorityQueue()

    # initialize an empty visited array
    visited = list()

    # initial_state is a tuple containing the start state coordinates, the sequence of 
    # actions taken till now to reach this state and the cost to reach that state from
    # the start state. initially the list of action will be empty as it remains empty.
    # and the initial cost will also be 0.
    initial_state = (problem.getStartState(), [], 0)

    # push the initial state in the queue
    priority_queue.push(initial_state, 0)

    # iterate over all the elements present in the queue until you reach the goal state
    # or the queue become empty.
    while(not priority_queue.isEmpty()):

        # pop the element from the queue, and do tuple unpacking to get the current 
        # state coordinates, the list of actions required to reach the current state and 
        # total cost required to reach.
        current_state, action_path, total_cost = priority_queue.pop()

        # if the current state is the goal state then return the list of actions.
        if(problem.isGoalState(current_state)):
            return action_path

        # if the state is not already visited. 
        if(current_state in visited):
            continue

        # add the states as a visited state.
        visited.append(current_state)

        # iterate over all it's neighbors and push those states in the queue.
        for next_state, action, cost in problem.getSuccessors(current_state):

            # add the action taken to reach the current state from the previous state
            # in the list of actions which contains the actions to reach the previous
            # state from the start state.
            next_path_action = action_path + [action]

            # Push the tuple in the stack. The tuple contains the coordinates of the 
            # next stage, the list of actions required to reach that stage and the total
            # cost required to reach. The priority will be the cost and the if the element
            # already present then update it's key value.
            priority_queue.update(tuple([next_state, next_path_action, cost + total_cost]), cost + total_cost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic = nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    # initialize an empty priority queue
    priority_queue = util.PriorityQueue()

    # initialize an empty visited array
    visited = list()

    # get the start state.
    start_state = problem.getStartState()

    #  initial_state is a tuple containing the start state coordinates, the sequence of 
    # actions taken till now to reach this state and the cost to reach that state from
    # the start state. initially the list of action will be empty as it remains empty.
    # and the initial cost will also be 0.
    initial_state = (start_state, [], 0)

    # push the initial state in the queue
    priority_queue.push(initial_state, heuristic(start_state, problem))

    # iterate over all the elements present in the queue until you reach the goal state
    # or the queue become empty.
    while(not priority_queue.isEmpty()):
        
        # pop the element from the queue, and do tuple unpacking to get the current 
        # state coordinates, the list of actions required to reach the current state and 
        # total cost required to reach.
        current_state, action_path, total_cost = priority_queue.pop()

        # if the current state is the goal state then return the list of actions.
        if(problem.isGoalState(current_state)):
            return action_path

        # if the state is not already visited. 
        if(current_state in visited):
            continue

        # add the states as a visited state.
        visited.append(current_state)

        # iterate over all it's neighbors and push those states in the queue.
        for next_state, action, cost in problem.getSuccessors(current_state):

            # add the action taken to reach the current state from the previous state
            # in the list of actions which contains the actions to reach the previous
            # state from the start state.
            next_path_action = action_path + [action]

            # calculate the estimated cost using the heuristic cost and the total cost.
            f_state = cost + total_cost + heuristic(next_state, problem)

            # Push the tuple in the stack. The tuple contains the coordinates of the 
            # next stage, the list of actions required to reach that stage and the total
            # cost required to reach. The priority will be the estimated cost and the if 
            # the element already present then update it's key value.
            priority_queue.update(
                tuple([next_state, next_path_action, cost + total_cost]), f_state)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
