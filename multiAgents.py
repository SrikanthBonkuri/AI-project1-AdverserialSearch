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
import math

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
        print(legalMoves, "\n")

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print(scores, "\n")
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        print("moves", legalMoves[chosenIndex])
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

        score = successorGameState.getScore()

        distanceToGhost1 = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if distanceToGhost1 > 0:
            score -= 30/distanceToGhost1

        if len(newGhostStates) > 1:
            distanceToGhost2 = manhattanDistance(newPos, newGhostStates[1].getPosition())
            if distanceToGhost2 > 0:
                score -= 30/distanceToGhost2

        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(distancesToFood):
            score += 10/min(distancesToFood)

        return score

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

    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0

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

        return self.minimax(gameState, 0, 0, 0)


    def minimax(self, gameState, depth, agent, root):
        if agent == gameState.getNumAgents():
            return self.minimax(gameState, depth+1, 0, root+1)   # here we reached all agents so, increment depth

        # check if we reached a terminal state
        if depth == self.depth or gameState.isWin() or gameState.isLose() or gameState.getLegalActions() == 0:
            return self.evaluationFunction(gameState)

        successors_score = []
        for action in gameState.getLegalActions(agent):
            successors_score.append(self.minimax(gameState.generateSuccessor(agent, action), depth, agent+1, root+1))

        if root == 0:
            actions = gameState.getLegalActions(agent)
            ind = successors_score.index(max(successors_score))
            return actions[ind]

        if self.isPacman(gameState, agent):
            return max(successors_score)
        else:
            return min(successors_score)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        return self.minimax(gameState, 0, 0, 0, -math.inf, math.inf)

    def minimax(self, gameState, depth, agent, root, alpha, beta):
        if agent == gameState.getNumAgents():
            return self.minimax(gameState, depth + 1, 0, root + 1, alpha, beta)  # here we reached all agents so, increment depth

        # check if we reached a terminal state
        if depth == self.depth or gameState.isWin() or gameState.isLose() or gameState.getLegalActions() == 0:
            return self.evaluationFunction(gameState)

        successors_score = []
        for action in gameState.getLegalActions(agent):
            score = self.minimax(gameState.generateSuccessor(agent, action), depth, agent + 1, root + 1, alpha, beta)
            successors_score.append(score)
            if self.isPacman(gameState, agent):
                if alpha < score:
                    alpha = score
            else:
                if beta > score:
                    beta = score

            if alpha > beta:
                break

        if root == 0:
            actions = gameState.getLegalActions(agent)
            ind = successors_score.index(max(successors_score))
            return actions[ind]

        if self.isPacman(gameState, agent):
            return max(successors_score)
        else:
            return min(successors_score)


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

        return self.minimax(gameState, 0, 0, 0)

    def minimax(self, gameState, depth, agent, root):
        if agent == gameState.getNumAgents():
            return self.minimax(gameState, depth + 1, 0, root + 1)  # here we reached all agents so, increment depth

        # check if we reached a terminal state
        if depth == self.depth or gameState.isWin() or gameState.isLose() or gameState.getLegalActions() == 0:
            return self.evaluationFunction(gameState)

        successors_score = []
        for action in gameState.getLegalActions(agent):
            successors_score.append(
                self.minimax(gameState.generateSuccessor(agent, action), depth, agent + 1, root + 1))

        if root == 0:
            actions = gameState.getLegalActions(agent)
            ind = successors_score.index(max(successors_score))
            return actions[ind]

        if self.isPacman(gameState, agent):
            return max(successors_score)
        else:
            prob = sum(successors_score)/len(successors_score)
            return prob


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:

    so here I implemented evaluation function by first taking the score of the current state, and iterating over all
    the new ghost states, and calculated the manhattan distance between current new state and the ghost.
    if the ghost is scared we can eat it so increase the score
    if the ghost is not scared we don't need to go near that so decrease the score
    now we need to check the distance between  current new state and all the foods, and add some score by considering the
    nearest food, like giving priority to the nearest food for the proposed state.

    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()

    for ghostState in newGhostStates:
        distanceToGhost = manhattanDistance(newPos, ghostState.getPosition())
        if distanceToGhost > 0:
            if ghostState.scaredTimer > 0:
                score += 30 / distanceToGhost
            else:
                score -= 30 / distanceToGhost

    distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
    if len(distancesToFood):
        score += 12 / min(distancesToFood)


    return score

# Abbreviation
better = betterEvaluationFunction