from NeuralNetwork import NeuralNetwork
from GeneticAlgorithm import Genetics
import gym
import numpy as np

env = gym.make('CartPole-v0')

number_of_inputs = env.observation_space.shape[0]
number_of_actions = env.action_space.n # number_of_outputs

size_of_network = [number_of_inputs, 4, 3, 1]

genetic = Genetics(size_of_network, 50, 0.01, 0.01, intervalo_geracao = 0.50)
network = NeuralNetwork( size_of_network, env = env, genetics = genetic)

the_best, _, n_episodes = network.learn()

sucess = network.test(the_best)

if sucess:
    print('Objetivo alcançado! \nForam necessários ', n_episodes, ' episódios para solucionar o desafio!')

input("Pressione ENTER para continuar")