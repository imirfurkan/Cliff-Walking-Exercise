# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 00:34:23 2023

@author: furkan
"""

import numpy as np
import seaborn as sns

class sarsa_learning():
    
    def __init__(self, env, alpha, gamma, epsilon, numberOfEpisodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberOfStates = env.observation_space.n
        self.numberOfActions = env.action_space.n
        self.numberOfEpisodes = numberOfEpisodes
        self.rewardList = np.zeros(numberOfEpisodes)
        self.Qmatrix = np.zeros((self.numberOfStates, self.numberOfActions)) # table for Q(s, a)
        
    def select_action(self, state): # using epsilon-greedy policy
        random_number = np.random.random() # [0, 1)
        
        if random_number < self.epsilon:
            return np.random.choice(self.numberOfActions)
        else:
            return np.random.choice(np.where(self.Qmatrix[state,:]==np.max(self.Qmatrix[state,:]))[0]) # [0] -> tuple to list
            #return np.argmax(self.Qmatrix[state, :]) # use the upper one in case there are multiple maximum choices
        
    def simulateEpisodes(self):
        
        for indexEpisode in range(self.numberOfEpisodes):
            
            currentState, info = self.env.reset()
            actionA = self.select_action(currentState)
            
            rewardEpisode = 0
            terminated = False
            while not terminated:
                newState, reward, terminated, _, _ = self.env.step(actionA)
                actionAprime = self.select_action(newState)
                rewardEpisode += reward
                
                if not terminated:
                    error = reward + self.gamma * self.Qmatrix[newState, actionAprime] - self.Qmatrix[currentState, actionA]
                    
                    self.Qmatrix[currentState, actionA] = self.Qmatrix[currentState, actionA] + self.alpha * error
                else:
                    error = reward - self.Qmatrix[currentState, actionA]
                    self.Qmatrix[currentState, actionA] = self.Qmatrix[currentState, actionA] + self.alpha * error
                
                currentState = newState
                actionA = actionAprime
            
            self.rewardList[indexEpisode] = rewardEpisode
    
    def get_rewards(self):
        return self.rewardList
    
    def get_Qmatrix(self):
        return self.Qmatrix
    
    def optimalQ(self):
        optimalQ = np.zeros((self.numberOfStates, 1))
        for stateS in range(int(self.numberOfStates)):
            optimalQ[stateS] = np.max(self.Qmatrix[stateS, :])
        optimalQ = optimalQ.reshape((4,12))
        return optimalQ
    
    def grid_policy(self):
        grid = np.zeros((self.numberOfStates, 1))
        for stateS in range(int(self.numberOfStates)):
            grid[stateS] = np.argmax(self.Qmatrix[stateS, :])
        grid = grid.reshape((4,12))
        return grid
    
class q_learning():

    def __init__(self, env, alpha, gamma, epsilon, numberOfEpisodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberOfStates = env.observation_space.n
        self.numberOfActions = env.action_space.n
        self.numberOfEpisodes = numberOfEpisodes
        self.rewardList = np.zeros(numberOfEpisodes)
        self.learnedPolicy = np.zeros(env.observation_space.n) # actions to be taken in certain states
        self.Qmatrix = np.zeros((self.numberOfStates, self.numberOfActions)) # table for Q(s, a)
    
    def select_action(self, state): # using epsilon-greedy policy
        random_number = np.random.random() # [0, 1)
        
        if random_number < self.epsilon:
            return np.random.choice(self.numberOfActions)
        else:
            return np.random.choice(np.where(self.Qmatrix[state,:]==np.max(self.Qmatrix[state,:]))[0]) # [0] -> tuple to list
            #return np.argmax(self.Qmatrix[state, :]) # use the upper one in case there are multiple maximum choices
        
    def simulateEpisodes(self):
        
        for indexEpisode in range(self.numberOfEpisodes):
            
            currentState, info = self.env.reset()
            
            rewardEpisode = 0
            terminated = False
            while not terminated:
                actionA = self.select_action(currentState)
                newState, reward, terminated, _, _ = self.env.step(actionA)
                rewardEpisode += reward
                
                if not terminated:
                    error = reward + self.gamma * np.max(self.Qmatrix[newState, :]) - self.Qmatrix[currentState, actionA]
                    self.Qmatrix[currentState, actionA] = self.Qmatrix[currentState, actionA] + self.alpha * error
                else:
                    error = reward - self.Qmatrix[currentState, actionA]
                    self.Qmatrix[currentState, actionA] = self.Qmatrix[currentState, actionA] + self.alpha * error
                
                currentState = newState
            
            self.rewardList[indexEpisode] = rewardEpisode
    
    def get_rewards(self):
        return self.rewardList
    
    def get_Qmatrix(self):
        return self.Qmatrix

    def optimalQ(self):
        optimalQ = np.zeros((self.numberOfStates, 1))
        for stateS in range(int(self.numberOfStates)):
            optimalQ[stateS] = np.max(self.Qmatrix[stateS, :])
        optimalQ = optimalQ.reshape((4,12))
        return optimalQ
    
    def grid_policy(self):
        grid = np.zeros((self.numberOfStates, 1))
        for stateS in range(int(self.numberOfStates)):
            grid[stateS] = np.argmax(self.Qmatrix[stateS, :])
        grid = grid.reshape((4,12))
        return grid