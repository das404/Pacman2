# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state extracting
    useful information for Q-learning algorithm
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        self.state = state
        self.key = self.convertKey(state)

    # Get the legal actions pacman can take
    def getLegalPacmanActions(self):
        return self.state.getLegalPacmanActions()

    # Given the state of class GameState convert it into a usable key for a dict classifying unique states
    # Includes pacman location, ghost location, and food locations
    def convertKey(self, state):
        foodLocations = []
        for i, foodList in enumerate(state.getFood()):
            for j, food in enumerate(foodList):
                if food == True:
                    foodLocations.append((i, j))
        return str(state.getPacmanPosition()) + str(state.getGhostPositions()[0]) + str(foodLocations)

    #Accessor Functions for state of class GameState and key of current game state
    def getState(self):
        return self.state

    def getKey(self):
        return self.key


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.1, # 0.2
                 epsilon: float = 0.05,
                 gamma: float = 0.6, # 0.05
                 maxAttempts: int = 2, # 5
                 numTraining: int = 10):

        """""
        Define Variables needed during Learning of Q-Learning Agent
        """""

        # Count the number of games we have played
        self.episodesSoFar = 0
        # Set up the qValue Action pair and Frequency action pair using the Counter class (dict)
        self.qValue = util.Counter()
        self.actionFrequency = util.Counter()
        # Set an optimisticReward that is used to force exploration
        self.OPTIMISTIC_REWARD = 1000
        # Initiate the lastAction variable to keep track of previous actions
        self.lastAction = Directions.STOP
        # Keep track of wins in training
        self.wins = 0
        # Initiate the startState variable (storing game state information of previous state)
        self.startState = None


        """
        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """

        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)

    # Accessor functions
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # Given a startState and an endState compute the reward as the difference in score
        return float(endState.getScore() - startState.getScore())


    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # Get the qValue of a corresponding state action pair using the transformed state key and the action as dictionary keys
        return self.qValue[(state.getKey(),action)]


    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Get the maximum qValue of all possible actions from the current state if there are actions possible from current state, otherwise return zero
        maxValues = []
        legalActions = state.getLegalPacmanActions()
        for action in legalActions:
            maxValues.append(self.qValue[(state.getKey(), action)])
        return max(maxValues) if legalActions else 0


    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Update the qValue of the initial state by incrementing it using the taken action,
        # the reward which the action achieved, and the maxQValue of the nextState
        self.qValue[(state.getKey(), action)] += self.getAlpha() * (
                    reward + self.getGamma() * self.maxQValue(nextState) - self.qValue[(state.getKey(), action)])


    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # Increment the frequency of the state action pair by 1
        self.actionFrequency[(state.getKey(), action)] += 1


    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        # Return the frequency of the state action pair
        return self.actionFrequency[(state.getKey(), action)]


    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # A way of implementing exploration
        # Return the optimistic reward if the frequency of a state action is below the threshold maxAttempts, forcing the agent tp pick this action
        # Else return the actual utility
        return float(self.OPTIMISTIC_REWARD) if counts < self.maxAttempts else utility


    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        Args:
            state: the current state

        Returns:
            The action to take
        """

        # Get the legal moves in the current state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Initiate wrapper class for current game state (endState)
        stateFeatures = GameStateFeatures(state)

        # Skip learning if the game just started and there is no last action
        if self.lastAction != Directions.STOP:
            # initiate wrapper class for previous game state (startState)
            startStateFeatures = GameStateFeatures(self.startState)
            # compute the reward as a difference in score between previous (startState) and current game state
            reward = self.computeReward(self.startState, state)
            # execute q-Learning given the previous state, current state, lastAction and reward
            self.learn(startStateFeatures, self.lastAction, reward, stateFeatures)
            # update the action frequency pair to keep track of actions already taken
            self.updateCount(startStateFeatures, self.lastAction)

        # shuffle the legal moves so that the action is chosen randomly if the utility values of actions are the same
        random.shuffle(legal)
        # Choose the action with the max qValue or explore if the count is below maxAttempts
        utility = []
        for action in legal:
            utility.append(self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action)))
        pick = legal[utility.index(max(utility))]

        # Store the previous state and action
        self.startState = state.deepCopy()
        self.lastAction = pick

        return pick

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """

        # Update the qValue and frequency one last time at the end of the game, to keep track of overall points obtained for win or loss
        startStateFeatures = GameStateFeatures(self.startState)
        stateFeatures = GameStateFeatures(state)
        reward = self.computeReward(self.startState, state)
        self.learn(startStateFeatures, self.lastAction, reward, stateFeatures)
        self.updateCount(startStateFeatures, self.lastAction)

        # Keep track of the number of wins in training
        if state.isWin():
            self.wins += 1

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            self.setAlpha(0)
            self.setEpsilon(0)
