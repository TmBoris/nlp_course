import pandas as pd
import numpy as np
import re
import email
from IPython.display import clear_output
from collections import defaultdict
from typing import Union, List


class NGramLanguageModel:
    def __init__(self, corpus, n):
        self.stats = {}
        self.n = n
        for text in corpus:
            for i in range(len(text) - self.n):
                if str(text[i:i + self.n]) not in self.stats:
                    self.stats[str(text[i:i + self.n])] = defaultdict(int)
                self.stats[str(text[i:i + self.n])][text[i + self.n]] += 1
                self.stats[str(text[i:i + self.n])]['1sum1'] += 1 # это чтобы преподсчитать сумму всех слов и потом поделить на нее. уникальность гарантируется наличием цифр в слове.


    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """
        if str(prefix[-self.n:]) not in self.stats:
            return [], []

        counter = self.stats[str(prefix[-self.n:])]['1sum1']
        dct = {k: (v / counter) for k, v in self.stats[str(prefix[-self.n:])].items() if k != '1sum1'}
        next_words, probs = list(dct.keys()), list(dct.values())

        return next_words, probs
