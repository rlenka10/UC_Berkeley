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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

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
"""        
class Node(object):
    def __init__(self, location, steps):
        self.location = location
        self.steps = steps
"""

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    "*** YOUR CODE HERE ***"    
    #startNode = [problem.getStartState(),[]]
    
    
    reached = set() #empty set for recording directions
    frontier = util.Stack() #dfs 
    startNode = (problem.getStartState(), []) #starting position with empty list
    
    frontier.push(startNode) #add startNode to frontier for traversal
    
    while not frontier.isEmpty(): 
        position, direction = frontier.pop() #pop nodes from frontier
        if (problem.isGoalState(position)): #if position == isGoalState
            return direction #return direction
        if (position not in reached): #if not
            reached.add(position) #add to reached set()
            for childLocation, step, stepCost in problem.getSuccessors(position): 
                childNode = (childLocation, direction + [step]) #get children
                frontier.push(childNode) #add to frontier for further traversal
                
                
                #newFrontier = list(node.actions) # destination pathway
                #newFrontier.append(children[1]) # add to list
                #frontier.push(startNode(children[0], newFrontier)) # add to end of array
                
            
            #childNode = problem.getSuccessors
            #for i in childNode:
                
    util.raiseNotDefined() # failure


    

"""    
function GRAPH-SEARCH(problem, frontier) return a solution or failure
    reached ← an empty set
    frontier← INSERT(MAKE-NODE(INITIAL-STATE[problem]), frontier)
    while not IS-EMPTY(frontier) do
        node ← POP(frontier)
        if problem.IS-GOAL(node.STATE) then return node
        end if
        if node.STATE is not in reached then
            add node.STATE in reached
            for each child-node in EXPAND(problem, node) do
                frontier ← INSERT(child-node, frontier)
            end for
        end if
    end while
    return failure    
"""  

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    reached = set() #empty set for recording directions
    frontier = util.Queue() #Queue for BFS 
    startNode = (problem.getStartState(), []) #starting position with empty list
    
    frontier.push(startNode) #add startNode to frontier for traversal
    
    while not frontier.isEmpty(): 
        position, direction = frontier.pop() #pop nodes from frontier
        if (problem.isGoalState(position)): #if position == isGoalState
            return direction #return direction
        if (position not in reached): #if not
            reached.add(position) #add to reached set()
            for childLocation, step, stepCost in problem.getSuccessors(position): 
                childNode = (childLocation, direction + [step]) #get children
                frontier.push(childNode) #add to frontier for further traversal
    
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    reached = set() #empty set for recording directions
    frontier = util.PriorityQueue() #PriorityQueue for UCS 
    startNode = (problem.getStartState(), 0, []) #starting position with empty list and 0 cost
    
    frontier.push(startNode, 0) #add startNode to frontier for traversal
    
    while not (frontier.isEmpty()):
        position, cost, direction = frontier.pop() #pop nodes from frontier
        if (problem.isGoalState(position)): #if position == isGoalState
            return direction #return direction
        if (position not in reached): #if not
            reached.add(position) #add to reached set()
            for childLocation, step, stepCost in problem.getSuccessors(position): 
                childStepCost = cost + stepCost #sum of previous cost and cost to reach child
                childNode = (childLocation, childStepCost, direction + [step]) #get children
                frontier.push(childNode, childStepCost) #return new child and updated cost
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    reached = set() # an empty set
    frontier = util.PriorityQueue() # Change to PriorityQueue for Uniform Cost Search
    startNode = (problem.getStartState(), 0, []) # starting position with addn cost var
    
    frontier.push(startNode, 0) # start traversal
    
    while not (frontier.isEmpty()): #while graph has nodes to expand
        position, cost, direction = frontier.pop() # remove node, addtn var for cost calc
        #currLocation = node.steps[-1]
        if (problem.isGoalState(position)): # if reached then return directions toward
            return direction # return steps taken to get to location
        if (position not in reached): # else run until empty
            reached.add(position) 
            # add current state to reached set as destination
            for childLocation, step, stepCost in problem.getSuccessors(position): 
            # expand fringe and record step cost
                childStepCost = cost + stepCost 
                # add starting cost to cost of steps taken
                childNode = (childLocation, childStepCost, direction + [step]) 
                # step must become list
                frontier.push(childNode, childStepCost + heuristic(childLocation, problem))
    
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
