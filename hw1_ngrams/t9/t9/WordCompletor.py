import pandas as pd
import numpy as np
import re
import email
from IPython.display import clear_output
from collections import defaultdict
from typing import Union, List


class PrefixTreeNode:
    def __init__(self):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()
        for word in vocabulary:
            cur_node = self.root
            for char in word:
                if char not in cur_node.children:
                    cur_node.children[char] = PrefixTreeNode()
                cur_node = cur_node.children[char]
            cur_node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """
        def _get_all_leaves(word, node, leaves):
            if node.is_end_of_word:
                leaves.append(word)

            for next_char, child in node.children.items():
                _get_all_leaves(word + next_char, child, leaves)

        start_node = self.root
        for char in prefix:
            if char not in start_node.children:
                return []
            start_node = start_node.children[char]

        leaves = []
        _get_all_leaves(prefix, start_node, leaves)
        return leaves


class WordCompletor:
    def __init__(self, corpus):
        """
        corpus: list – корпус текстов
        """
        self.word_counter = 0
        self.word_to_num = defaultdict(int)
        for text in corpus:
            for word in text:
                self.word_to_num[word] += 1
                self.word_counter += 1
        self.prefix_tree = PrefixTree(self.word_to_num.keys())

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words, probs = [], []
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.word_to_num[word] / self.word_counter for word in words]
        return words, probs
