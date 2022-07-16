
def default(x, default_val):
    if x is None: return default_val
    return x

def default_token(list_, index):
    if index == -1:
        return "<start>"
    if index == len(list_):
        return "<end>"
    return list_[index]

class QLeaarning:
    def __init__(self):
        pass

import re
import collections

class TextAnalysis:
    def __init__(self, text: str):
        self.text = text
        self.words = [i for i in re.split(r"[,\W.;:]", text) if i != '']
        self.len = len(self.words)
        self.bigrams = [(default_token(self.words, i), default_token(self.words, i + 1)) for i in range(-1, self.len)]
        self.occurances = collections.Counter(self.words)
        self.bigram_occurances = collections.Counter(self.bigrams)
        self.vocab = len(self.occurances)

    def unigram(self, string: str):
        return default(self.occurances.get(string), 0) / self.len

    def all_unigrams(self):
        return [(i, self.unigram(i)) for i in self.occurances]

    def unigram_add_one(self, string: str):
        return (1 + default(self.occurances.get(string), 0)) / (self.len + self.vocab)
            
    def bigram(self, strleft: str, strright: str):
        return default(self.bigram_occurances.get((strleft, strright)), 0) / self.occurances.get(strleft)
        
t = TextAnalysis("""Laterne, Laterne,
        Sonne, Mond und Sterne.
        Brenne auf mein Licht,
        brenne auf mein Licht,
        aber nur meine liebe Laterne nicht""")

print(t.words, t.unigram("Laterne"))
print(t.all_unigrams())
print(t.bigram_occurances)
print(t.bigram("auf", "mein"))




