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

        foodLocations = currentGameState.getFood().asList()
        distanceFood = []
        distanceGhost = []

        if action == "Stop":
            return -999

        for foodLocation in foodLocations:
            x = -abs(newPos[0] - foodLocation[0])
            y = -abs(newPos[1] - foodLocation[1])
            distanceFood.append(x + y)
        for ghostState in newGhostStates:
            x = abs(newPos[0] - ghostState.getPosition()[0])
            y = abs(newPos[1] - ghostState.getPosition()[1])
            if (x + y < 2 and ghostState.scaredTimer is 0):
                return -999
            if (ghostState.scaredTimer > 0 and x + y < 10):
                return 999

        "*** YOUR CODE HERE ***"
        return max(distanceFood)

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
        "*** YOUR CODE HERE ***"
        return minimax(gameState, self.index, 0)

        def minimax(self, state, agent, depth):
            successor = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]

            if successor == []:
                return self.evaluationFunction(state)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.index, 0, -99999, 99999)[1]
    def minimax(self, state, agent, depth, alpha, beta):
        if agent % state.getNumAgents() == 0:
            if depth == self.depth:
                return self.evaluationFunction(state), None
            return self.maximizer(state, agent % state.getNumAgents(), depth, alpha, beta)
        return self.minimizer(state, agent % state.getNumAgents(), depth, alpha, beta)

    def minimizer(self, state, agent, depth, alpha, beta):
        if len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None


        value = 99999; o_action = None;
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)

            n_value, n_action = self.minimax(successor, agent + 1, depth, alpha, beta)
            if n_value < value:
                value = n_value
                action = o_action
            if value < alpha:
                return value, o_action
            beta = min(beta, value)
        return value, o_action

    def maximizer(self, state, agent, depth, alpha, beta):
        if len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        value = -99999; o_action = None;
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            n_value, n_action = self.minimax(successor, agent + 1, depth + 1, alpha, beta)
            if n_value > value:
                value = n_value
                o_action = action
            if value > beta:
                return value, o_action
            alpha = max(alpha, value)
        return value, o_action

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
        "*** YOUR CODE HERE ***"
        scores = []
        def expectiMax(self, state, agent, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            score = []
            for action in state.getLegalActions(agent):
                if (agent ==  gameState.getNumAgents() - 1):
                    if(depth == 1):
                        score.append(self.evaluationFunction(state.generateSuccessor(agent, action)))
                    else:
                        score.append(expectiMax(self, state.generateSuccessor(agent, action), agent, depth - 1))
                else:
                    score.append(expectiMax(self, state.generateSuccessor(agent, action), agent + 1, depth))
            if agent == 0:
                return max(score)
            else:
                return sum(score) / len(score)

        for action in gameState.getLegalActions(0):
            scores.append(expectiMax(self, gameState.generateSuccessor(0, action), 1, self.depth))
        return gameState.getLegalActions(0)[scores.index(max(scores))]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()# multiAgents.py
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

        foodLocations = currentGameState.getFood().asList()
        distanceFood = []
        distanceGhost = []

        if action == "Stop":
            return -999

        for foodLocation in foodLocations:
            x = -abs(newPos[0] - foodLocation[0])
            y = -abs(newPos[1] - foodLocation[1])
            distanceFood.append(x + y)
        for ghostState in newGhostStates:
            x = abs(newPos[0] - ghostState.getPosition()[0])
            y = abs(newPos[1] - ghostState.getPosition()[1])
            if (x + y < 2 and ghostState.scaredTimer is 0):
                return -999
            if (ghostState.scaredTimer > 0 and x + y < 10):
                return 999

        "*** YOUR CODE HERE ***"
        return max(distanceFood)

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
        "*** YOUR CODE HERE ***"

        def minimax(state, depth, agent):
            score = []

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                if (agent == gameState.getNumAgents() - 1):
                    if(depth == 1):
                        score.append(self.evaluationFunction(state.generateSuccessor(agent, action)))
                    else:
                        score.append(minimax(state.generateSuccessor(agent, action), depth - 1, 0))
                else:
                    score.append(minimax(state.generateSuccessor(agent, action), depth, agent + 1))

            if agent == 0:
                return max(score)
            else:
                return min(score)

        scores = []
        for action in gameState.getLegalActions(0):
            scores.append(minimax(gameState.generateSuccessor(0, action), self.depth, 1))
        return gameState.getLegalActions(0)[scores.index(max(scores))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.index, 0, -99999, 99999)[1]
    def minimax(self, state, agent, depth, alpha, beta):
        if agent % state.getNumAgents() == 0:
            if depth == self.depth:
                return self.evaluationFunction(state), None
            return self.maximizer(state, agent % state.getNumAgents(), depth, alpha, beta)
        return self.minimizer(state, agent % state.getNumAgents(), depth, alpha, beta)

    def minimizer(self, state, agent, depth, alpha, beta):
        if len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None


        value = 99999; o_action = None;
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)

            n_value, n_action = self.minimax(successor, agent + 1, depth, alpha, beta)
            if n_value < value:
                value = n_value
                action = o_action
            if value < alpha:
                return value, o_action
            beta = min(beta, value)
        return value, o_action

    def maximizer(self, state, agent, depth, alpha, beta):
        if len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        value = -99999; o_action = None;
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            n_value, n_action = self.minimax(successor, agent + 1, depth + 1, alpha, beta)
            if n_value > value:
                value = n_value
                o_action = action
            if value > beta:
                return value, o_action
            alpha = max(alpha, value)
        return value, o_action

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
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agent):
          score = []
          if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

          for action in  state.getLegalActions(agent):
            if (agent == gameState.getNumAgents()-1):
                if(depth == 1 ):
                    score.append(self.evaluationFunction(state.generateSuccessor(agent, action)))
                else:
                    score.append(expectimax(state.generateSuccessor(agent, action),depth-1,0))
            else:
              score.append(expectimax(state.generateSuccessor(agent, action),depth,agent+1))

          if agent == 0:
            return max(score)
          else:
            return sum(score) / len(score)

        scores = []
        for action in gameState.getLegalActions(0):
          scores.append(expectimax(gameState.generateSuccessor(0, action), self.depth, 1))

        return gameState.getLegalActions(0)[scores.index(max(scores))]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    posPacman = currentGameState.getPacmanPosition()
    stateGhost = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    score = 0

    for ghost in stateGhost:
        dist = manhattanDistance(posPacman, ghost.getPosition())
        if dist > 11 and ghost.scaredTimer > 0:
            score -= dist * 50
        else:
            score += dist * 30
    score -= food.count() * 2
    return score
# Abbreviation
better = betterEvaluationFunction


# Abbreviation
better = betterEvaluationFunction
