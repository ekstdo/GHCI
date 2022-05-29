# Hier ist einfach nur eine Implementation von paar Algorithmen,
# die in GHCI vorgestellt worden sind

from typing import Any
from pandas import DataFrame
import numpy as np
import random
import copy
from collections import Counter

def echo(*args): 
    print(*args)
    return args[-1]

class NaiveBayes:
    table: DataFrame

    def __init__(self, table: DataFrame):
        self.table = table

    def evaluate(self, query: str, value: Any, attributes: list[Any], m: int) -> float:
        resulting_frame = self.table[self.table[query] == value]
        queried_len = resulting_frame.shape[0]
        value = queried_len / self.table.shape[0]

        for i, j in enumerate(self.table.columns):
            if j == query:
                continue

            sub_frame = resulting_frame[resulting_frame[j] == attributes[i]]
            sub_len = sub_frame.shape[0]
            if sub_len == 0:
                value *= m / self.table[j].nunique() / (m + queried_len)
            else:
                value *= sub_len / queried_len
        return value

    def interpret(self, query: str, attributes: list[Any], m: int, verbose: bool = False):
        resulting_value = None
        max_value = 0
        for i in self.table[query].unique():
            result = self.evaluate(query, i, attributes, m)
            if verbose:
                print(f"{i}: {result}")
            if result > max_value:
                max_value = result
                resulting_value = i

        return resulting_value

data = DataFrame(
        [["Jenaplan", "kooperativ", "emotional", "positiv"], 
        ["Montessori", "kooperativ", "emotional", "negativ"], 
        ["Waldorf", "kompetitiv", "emotional", "positiv"], 
        ["Jenaplan", "kompetitiv", "emotional", "positiv"], 
        ["Montessori", "kompetitiv", "rational", "negativ"], 
        ["Montessori", "kompetitiv", "emotional", "negativ"], 
        ["Jenaplan", "kompetitiv", "rational", "negativ"], 
        ["Montessori", "kooperativ", "emotional", "negativ"], 
        ["Montessori", "kooperativ", "emotional", "negativ"], 
        ["Jenaplan", "kompetitiv", "emotional", "negativ"], 
        ["Montessori", "kompetitiv", "rational", "negativ"], 
        ["Waldorf", "kooperativ", "emotional", "positiv"]], columns=["Pädagogik", "Verhalten", "Charakter", "Erfahrung"])


nb = NaiveBayes(data)

# nb.evaluate("Erfahrung", "positiv", ["Jenaplan", "kooperativ", "rational"], 12)
# print(nb.interpret("Erfahrung", ["Jenaplan", "kooperativ", "rational"], 12, True))


class HiddenMarkovModell:
    A: np.ndarray
    initial: np.ndarray
    observation: np.ndarray
    
    def __init__(self, transition: np.ndarray, initial: np.ndarray, observation: np.ndarray):
        self.A = transition
        self.initial = initial
        self.observation = observation

    def evaluate(self, observations: np.ndarray, t: int = -1, verbose = False):
        if t == -1:
            t = len(observations) - 1
        cache = {}
        ret = sum(self.evaluate_alpha(observations, t, i, cache) for i in range(len(self.A)))
        if verbose:
            print(cache)
        return ret

    def evaluate_alpha(self, observations: np.ndarray, t: int, state: int, cache = {}):
        results = cache
        def alpha(t: int, state: int):
            if (t, state) in results:
                return results[(t, state)]

            if t == 0:
                res = self.initial[state] * self.observation[state][observations[0]]
                results[(t, state)] = res
                return res
            res = sum(alpha(t - 1, i) * self.A[i][state]
                    for i in range(len(self.A))) * self.observation[state][observations[t]]
            results[(t, state)] = res
            return res

        return alpha(t, state)

    def back_evaluate(self, observations: np.ndarray, t: int = 0, T: int = -1, verbose = False):
        cache = {}
        ret = sum(self.evaluate_beta(observations, t, i, T, cache) * self.observation[i][observations[0]] * self.initial[i] for i in range(len(self.A)))
        if verbose:
            print(cache)
        return ret

    def evaluate_beta(self, observations: np.ndarray, t: int, state: int, T: int = -1, cache = {}):
        if T == -1: T = len(observations) - 1

        results = cache
        def beta(t: int, state: int):
            if (t, state) in results:
                return results[(t, state)]

            if t == T:
                results[(t, state)] = 1.0
                return 1.0

            res = sum(self.A[state][j] * self.observation[j][observations[t + 1]] * beta(t + 1, j) 
                    for j in range(len(self.A)))
            results[(t, state)] = res
            return res
        return beta(t, state)

    def interpret(self, observations: np.ndarray, t: int):
        results = np.full((t + 1, len(self.A)), -1.)
        def interpret_(t: int, state: int):
            if results[t][state] != -1:
                return results[t][state]
            if t == 0:
                res = self.initial[state] * self.observation[state][observations[0]]
                results[t][state] = res
                return res
            res = max(interpret_(t - 1, i) * self.A[i][state]
                    for i in range(len(self.A))) * self.observation[state][observations[t]]
            results[t][state] = res
            return res

        for i in range(len(self.A)):
            interpret_(t, i)

        observation_chain = []
        for i in results:
            observation_chain.append(np.argmax(i))

        return observation_chain

    def train(self, observations: np.ndarray):
        alpha_cache = {}
        beta_cache = {}

        evaluated = self.evaluate(observations)
        l = len(observations)

        def get_alpha(t, i):
            if (t, i) in alpha_cache:
                return alpha_cache[(t, i)]

            return self.evaluate_alpha(observations, t, i, alpha_cache)

        def get_beta(t, i):
            if (t, i) in beta_cache:
                return beta_cache[(t, i)]


            return self.evaluate_beta(observations, t, i, len(observations) - 1, beta_cache)


        def gamma(t, i):
            return get_alpha(t, i) * get_beta(t, i) / evaluated

        def xi(i, j, t):
            return get_alpha(t, i) * get_beta(t + 1, i) * self.A[i][j] * self.observation[j][observations[t + 1]] / evaluated

        new_A = np.zeros_like(self.A)
        new_observation = np.zeros_like(self.observation)
        new_initial = np.zeros_like(self.initial)

        for i in range(len(self.A)):
            for j in range(len(self.A[0])):
                new_A[i][j] = sum(xi(i, j, t) for t in range(l - 1)) / sum(gamma(t, i) for t in range(l - 1))
            for j in range(len(self.observation[0])):
                new_observation[i][j] = sum(gamma(t, i) for t in range(l) if j == observations[t]) /  sum(gamma(t, i) for t in range(l))

            new_initial[i] = gamma(0, i)

        self.A = new_A
        self.observation = new_observation
        
hmm = HiddenMarkovModell(np.array([[0.6, 0.4], [0.0, 1.0]]),
                        np.array([1.0, 0.0]),
                        np.array([[0.8, 0.2], [0.3, 0.7]]))

# print(hmm.evaluate(np.array([0, 0]), 1, 0))
# print(hmm.evaluate(np.array([0, 0, 1]), 2, 0))
# print(hmm.back_evaluate(np.array([0, 0, 1]), 0, 0))
# print(hmm.interpret(np.array([0, 0, 1]), 2))


# hmm2 = HiddenMarkovModell(np.array([[0.5, 0.5], [0.5, 0.5]]),
#     np.array([1.0, 0.0]),
#     np.array([[0.8, 0.2], [0.2, 0.8]]))

# print(hmm2.evaluate(np.array([0, 0, 1]), 2, 0))
# print(hmm2.back_evaluate(np.array([0, 0, 1]), 0, 0))
# print(hmm2.interpret(np.array([0, 0, 1]), 2))
# for i in range(7):
#     print(f"after: {i} Iterations:", hmm2.A)
#     hmm2.train(np.array([0, 1, 1]))


# hmm3 = HiddenMarkovModell(np.array([[0.0, 1.0], [0.0, 1.0]]),
#     np.array([1.0, 0.0]),
#     np.array([[1.0, 0.0], [0.0, 1.0]]))


# print(hmm3.back_evaluate(np.array([0, 1, 1]), 0))
# print(hmm3.evaluate(np.array([0, 1, 1])))
# print(hmm3.interpret(np.array([0, 1, 1]), 2))
# print("before: ", hmm3.A)
# hmm3.train(np.array([0, 1, 1]))
# print("after: ", hmm3.A)

def get_list_type(list_):
    if len(list_) == 0:
        return Any
    current_type = type(list_[0])
    if all(type(i) == current_type for i in list_):
        return current_type
    return Any

def ultimate_equal(x, y):
    return len(x) == len(y) and all(len(x[i]) == len(y[i]) and np.array_equal(x[i], y[i]) for i in range(len(x)))


class KMeans:
    def __init__(self, data: list[Any]):
        self.data = np.array(data, dtype=float)
        self.centers = np.array([])

    def init(self, k: int):
        # Initialize
        mask = np.zeros(len(self.data), dtype= bool)

        for i in np.random.choice(len(self.data), k, replace = False):
            mask[i] = True
        
        copy = self.data[mask, ...]
        self.clusters = [[np.copy(i)] for i in copy]
        self.centers = copy

        mask = np.logical_not(mask)

        # Einordnung
        for i in self.data[mask, ...]:
            distances = np.linalg.norm(np.subtract(self.centers, i), axis= 1)
            index = np.argmin(distances)
            self.clusters[index].append(i)

        # Berechnung
        for (index, i) in enumerate(self.clusters):
            self.centers[index] = sum(i) / len(i)


    def iteration(self, k: int):
        self.clusters = [[] for i in range(k)]

        # Einordnung
        for i in self.data:
            distances = np.linalg.norm(np.subtract(self.centers, i), axis= 1)
            index = np.argmin(distances)
            self.clusters[index].append(i)

        # Berechnung
        for (index, i) in enumerate(self.clusters):
            self.centers[index] = sum(i) / len(i)

    def run(self, k: int, verbose = False, init = True):
        if init:
            self.init(k)
        last_clusters = copy.deepcopy(self.clusters)

        self.iteration(k)
        current_clusters = copy.deepcopy(self.clusters)
        i = 1
        while not ultimate_equal(last_clusters, current_clusters):
            last_clusters = current_clusters
            if verbose:
                print(current_clusters, self.centers)
            self.iteration(k)
            current_clusters = copy.deepcopy(self.clusters)
            i += 1
        if verbose: 
            print(current_clusters, self.centers)
            print(f"{i} Durchläufe")


km = KMeans([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
km.run(3, True)


class KNearest:
    def __init__(self, clusters):
        self.clusters = clusters

    def run(self, val, k: int, verbose = False) -> int:
        c = [sorted(((j, np.linalg.norm(j - val)) for j in i), reverse=True, key= lambda x: x[1]) for i in self.clusters]
        result = []
        counter = Counter()
        for i in range(k):
            index = min(range(len(c)), key = lambda x:  c[x][-1][1] if len(c[x]) > 0 else float("inf"))
            result.append((index, c[index].pop()))
            if verbose:
                print(index, result)

        for i in result:
            if i[0] in counter:
                counter[i[0]] += 1
            else:
                counter[i[0]] = 1

        return counter.most_common(1)[0][0]
                




kn = KNearest(km.clusters)
print(kn.run(np.array([2, 5]), 3, True))

def relu(x):
    x[x < 0] = 0

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class NeuralNetwork:
    weights: list[np.ndarray]
    bias: list[np.ndarray]

    def __init__(self, weights: list[np.ndarray] = [], bias: list[np.ndarray] = [], activation_functions = None):
        self.weights = weights
        self.bias = bias
        if type(activation_functions) is list:
            self.activation_functions = activation_functions
        elif activation_functions is None:
            self.activation_functions = [relu for i in range(len(self.bias))]
        else:
            self.activation_functions = [activation_functions for i in range(len(self.bias))]


    def random_init(self, shape: list[int]):
        for i in range(len(shape) - 1):
            self.weights.append(np.random.random_sample((shape[i], shape[i + 1])))
            self.bias.append(np.random.random_sample((shape[i + 1], )))

    def evaluate(self, input_: np.ndarray, verbose = False):
        between = input_
        for i in range(len(self.weights)):
            between = np.matmul(between, self.weights[i]) + self.bias[i]
            res = self.activation_functions[i](between)
            if res is not None:
                between = res
            if verbose:
                print(f"Layer {i}: {between}")
        return between

    def backpropagation(self, data, should_result, is_result):
        dif_weights = []
        dif_bias = []

        delta_neuron = [is_result * (1 - is_result) * (should_result - is_result)]

        for i in range(len(self.bias) - 1):
            delta_neuron.append(is_result * (1 - is_result) * sum(delta_neuron[k] * self.weights[k] for k in range(len(self.bias[len(self.bias) - 1 - i]))))

        print(delta_neuron)


# Perzeptron

# nn = NeuralNetwork([np.array([0.1, .1])], [np.array([0])])

# print(nn.evaluate(np.array([0., 0.])))
# print(nn.evaluate(np.array([0., 1.])))
# print(nn.evaluate(np.array([1., 0.])))
# print(nn.evaluate(np.array([1., 1.])))


nn2 = NeuralNetwork([
    np.array([[0.5, 0.9], [0.4, 1.0]]), # weights layer 1
    np.array([-1.2, 1.1])], # weights layer 2

    [np.array([-0.8, 0.1]), # bias layer 1
        np.array([-0.3])], sigmoid)

print(nn2.evaluate(np.array([1., 1.]), True))
