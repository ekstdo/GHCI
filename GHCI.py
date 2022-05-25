# Hier ist einfach nur eine Implementation von paar Algorithmen,
# die in GHCI vorgestellt worden sind

from typing import Any
from pandas import DataFrame
import numpy as np

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


hmm2 = HiddenMarkovModell(np.array([[0.5, 0.5], [0.5, 0.5]]),
    np.array([1.0, 0.0]),
    np.array([[0.8, 0.2], [0.2, 0.8]]))

print(hmm2.evaluate(np.array([0, 0, 1]), 2, 0))
print(hmm2.back_evaluate(np.array([0, 0, 1]), 0, 0))
print(hmm2.interpret(np.array([0, 0, 1]), 2))
for i in range(7):
    print(f"after: {i} Iterations:", hmm2.A)
    hmm2.train(np.array([0, 1, 1]))


# hmm3 = HiddenMarkovModell(np.array([[0.0, 1.0], [0.0, 1.0]]),
#     np.array([1.0, 0.0]),
#     np.array([[1.0, 0.0], [0.0, 1.0]]))


# print(hmm3.back_evaluate(np.array([0, 1, 1]), 0))
# print(hmm3.evaluate(np.array([0, 1, 1])))
# print(hmm3.interpret(np.array([0, 1, 1]), 2))
# print("before: ", hmm3.A)
# hmm3.train(np.array([0, 1, 1]))
# print("after: ", hmm3.A)
