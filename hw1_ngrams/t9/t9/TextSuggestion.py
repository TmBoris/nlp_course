import pandas as pd
import numpy as np
import re
import email
from IPython.display import clear_output
from collections import defaultdict
from typing import Union, List
import heapq


class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(self, text: str, n_words=3, n_texts=1, beam_width=1) -> list[str]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)
        
        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)
        
        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """
        if len(text) == 0:
            return ['']

        suggestions = []
        text_wo_last_word = ' '.join(text.split()[:-1])
        last_word = text.split()[-1]
        
        words, word_probs = self.word_completor.get_words_and_probs(last_word)
        if len(word_probs) == 0:
            return ['']
        
        word_probs_inds = np.argsort(word_probs)[-2:]
        beam = []
        for i in word_probs_inds:
            beam.append((word_probs[i], text_wo_last_word + ' ' + words[i]))

        # print('predict_to_proba after word completor:')
        # print('\t', beam)

        for _ in range(n_words):
            all_candidates = []
            for proba, cur_predict in beam:
                # print('next element from beam')
                # print('\tcur_predict:', cur_predict)
                # print('\tproba:', proba)

                next_words, next_word_probs = self.n_gram_model.get_next_words_and_probs(cur_predict.split())
                # print('next_word_probs:', next_word_probs)
                if len(next_word_probs) == 0:
                    continue
                sorted_probs_inds = np.argsort(next_word_probs)[-2:]

                for i in sorted_probs_inds:
                    all_candidates.append((proba + next_word_probs[i], cur_predict + ' ' + next_words[i]))
            
            beam = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[0])

            
            print(beam)

        for i, pair in enumerate(beam):
            prob, suggestion = pair
            if i == n_texts:
                break
            suggestions.append(suggestion)

        # тут нужно посортить pridict_to_proba по значениям и выбрать наиболее вероятностные предложения
        # мб забабахать эвристику тип если нет ни у чего вероятности хотя бы сколько-то то скипы
        # print(f'collected {len(suggestions)} suggestions')
        return suggestions
    