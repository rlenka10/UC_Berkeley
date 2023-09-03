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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # calc manhattan dist between Pacman curr pos and remaining ghost pos and store in list
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        # calc manhattan dist between Pacman curr pos and remaining food pos and store in list
        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        # avg ghost distances, if empty = 0
        avg_gd = sum(ghost_distances) / len(ghost_distances) if ghost_distances else 0
        # find min value of food_distances, if empty = 0
        md_score = min(food_distances) if food_distances else 0

        # returns sum of curr score, min scared time
        return successorGameState.getScore() + min(newScaredTimes) + 1 / (md_score + 0.1) - 1 / (avg_gd + 0.1)

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        return self.minimax(0, 0, gameState)[0]

    def minimax(self, depth, agentIndex, state):

        # Check if depth equals the self defined depth, or the game has been won or lost
        # If so, return None for the best action and the evaluation of the current state
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        # Define two variables named topScore and topAction with None value
        topScore, topAction = None, None
        
        # Loop through all legal actions of the current agent
        for action in state.getLegalActions(agentIndex):
            
            # Generate the successor state after the current agent takes the current action
            nState = state.generateSuccessor(agentIndex, action)
            
            # Get the index of the next agent in the game
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            # Call the minimax function with the new depth, the next agent's index, and the new state
            # The resulting score is saved in the variable "score"
            _, score = self.minimax(depth + (nextAgentIndex == 0), nextAgentIndex, nState)
            
            # If the current agent is Pacman and the score is greater than the current best score or
            # the current agent is a ghost and the score is less than the current best score
            # then update the best score and best action with the current score and action
            if (agentIndex == 0 and (topScore is None or score > topScore)) \
                    or (agentIndex != 0 and (topScore is None or score < topScore)):
                topScore = score
                topAction = action

        # Return the best action and the best score
        return topAction, topScore
    
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta_pruning(state, agent, depth, alpha, beta):
            # Check if the game is over or we have reached the maximum depth
            if state.isWin() or state.isLose() or depth == self.depth:
                # Return the current score of the game and no action
                return self.evaluationFunction(state), None
            
            # Check if it's max player's turn
            if agent == 0:
                # Init the value of the best action to negative infinity
                value = float('-inf')
                for action in state.getLegalActions(agent):
                    # Generate a successor state for the current action
                    successor = state.generateSuccessor(agent, action)
                    # Recursively get the value of the successor state for the next player (min player)
                    successor_value, _ = alpha_beta_pruning(successor, 1, depth, alpha, beta)
                    # Update the value of the best action if the successor state has a higher value
                    if successor_value > value:
                        value = successor_value
                        topAction = action
                    # If the value of the successor state is greater than beta, then we can prune the rest of the tree
                    if value > beta:
                        return value, topAction
                    # Update alpha to the maximum value seen so far
                    alpha = max(alpha, value)
                # Return the value of the best action and the best action itself
                return value, topAction
            else:
                # Initialize the value of the best action to positive infinity
                value = float('inf')
                for action in state.getLegalActions(agent):
                    # Generate a successor state for the current action
                    successor = state.generateSuccessor(agent, action)
                    # Recursively get the value of the successor state for the next player
                    successor_value, _ = alpha_beta_pruning(successor, (agent + 1) % state.getNumAgents(), depth + (agent + 1 == state.getNumAgents()), alpha, beta)
                    # Update the value of the best action if the successor state has a lower value
                    value = min(value, successor_value)
                    # If the value of the successor state is less than alpha, then we can prune the rest of the tree
                    if value < alpha:
                        return value, None
                    # Update beta to the minimum value seen so far
                    beta = min(beta, value)
                # Return the value of the best action and no action
                return value, None

        # Call the alpha-beta pruning function to get the best action for pacman (max player)
        _, action = alpha_beta_pruning(gameState, 0, 0, float('-inf'), float('inf'))
        # Return the best action for pacman to perform
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.expectimax(0, 0, gameState)  
        return action  

    # define recursive method to return best actions and scores
    def expectimax(self, curr_depth, agentIndex, gameState):
        # check if index is greater than or equal to total number of agents
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            curr_depth += 1
        # check if curr depth is
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)

        # initialize best score and best action as None
        topScore, topAction = None, None

        # player turn
        if agentIndex == 0: 
            # loop over all legal actions of player
            for action in gameState.getLegalActions(agentIndex):
                # recursively call expectimax with next game state and agent index incremented by 1
                _, score = self.expectimax(curr_depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, action))
                # if the score is greater than the best score so far, update the best score and action
                if topScore is None or score > topScore:
                    topScore = score
                    topAction = action
        # ghost turn
        else:
            # get legal actions of ghost
            ghostActions = gameState.getLegalActions(agentIndex)
            if ghostActions:
                # calc the prob of each legal action of ghost agent
                prob = 1.0 / len(ghostActions)
                # loop over all legal actions of ghost agent
                for action in ghostActions:
                    # call expectimax
                    _, score = self.expectimax(curr_depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, action))
                    # init best score if not init
                    if topScore is None:
                        topScore = 0.0
                        # update best score
                    topScore += prob * score
                    topAction = action

                    # return best action
        return (topAction, self.evaluationFunction(gameState)) if topScore is None else (topAction, topScore)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Python function that takes in Gamestate argument and returns score.
    Heuristic evaluation function
    """
    # Initialize the eval score to the current game state score
    evaluation = currentGameState.getScore()

    # If the game is won, return positive infinity
    if currentGameState.isWin():
        return float('inf')

    # Get the current position of Pacman
    newPos = currentGameState.getPacmanPosition()

    # Initialize the ghost penalty to 0
    gdPenalty = 0

    # Get a list of all the ghost states in the game
    ghostStates = currentGameState.getGhostStates()

    # Get the number of ghosts in the game
    numGhosts = len(ghostStates)

    # Initialize the ghost index to 0
    ghostIndex = 0

    # Iterate over every ghost in the game
    while ghostIndex < numGhosts:
        # Get the state and position of the current ghost
        ghostState = ghostStates[ghostIndex]
        ghostPos = ghostState.configuration.getPosition()

        # Calculate the Manhattan distance between Pacman and the ghost
        distance_to_ghost = manhattanDistance(ghostPos, newPos)

        # If the ghost is not scared
        if ghostState.scaredTimer <= 1:
            # Penalize Pacman if he is too close to the ghost
            if distance_to_ghost < 4:
                gdPenalty -= (10 - distance_to_ghost)
            # If Pacman is too close to a non-scared ghost, return negative infinity
            elif distance_to_ghost < 2:
                return -float('inf')
        # Increment the ghost index
        ghostIndex += 1

    # Subtract  Manhattan distance between Pacman and each capsule from the eval score
    for capsule in currentGameState.getCapsules():
        evaluation -= manhattanDistance(capsule, newPos)

    # Initialize the closest food distance to infinity
    closest_food_distance = float('inf')

    # Initialize the sum of closest food distances to 0
    sum_closest_food_distance = 0

    # Get the number of remaining food pellets in the game
    num_food = currentGameState.getNumFood()

    # Iterate over every food pellet in the game
    for food in currentGameState.getFood().asList():
        # Calculate the Manhattan distance between Pacman and the food pellet
        distance_to_food = manhattanDistance(food, newPos)

        # Update the closest food distance if necessary
        closest_food_distance = min(closest_food_distance, distance_to_food)

        # Add the negative of the distance to the sum of closest food distances
        sum_closest_food_distance -= distance_to_food / 10

    # Subtract the closest food distance, the number of remaining food pellets times 5, the sum of closest food distances, and the ghost penalty from the eval score
    evaluation -= closest_food_distance + num_food * 5 + sum_closest_food_distance + gdPenalty

    # Return the final eval score
    return evaluation

# Abbreviation
better = betterEvaluationFunction
