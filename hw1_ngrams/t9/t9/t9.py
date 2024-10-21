"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import pandas as pd
import numpy as np
import re
import email

from rxconfig import config
from IPython.display import clear_output
from collections import defaultdict
from typing import Union, List

from .dataPreparation import body, tokenize
from .WordCompletor import WordCompletor
from .nGramModel import NGramLanguageModel
from .TextSuggestion import TextSuggestion


emails = pd.read_csv('t9/emails.csv')
emails['body'] = body(emails['message'])
tok_emails = [tokenize(mail) for mail in emails['body']]

word_completor = WordCompletor(tok_emails)
n_gram_model = NGramLanguageModel(corpus=tok_emails, n=2)
text_suggestion = TextSuggestion(word_completor, n_gram_model)

class State(rx.State):
    """The app state."""
    current_input: str = ""
    suggestions = ['']
    
    def update_suggestions(self, current_input: str):
        """
        :param: **current_input** - текст на данный момент в окошке ввода.
        """
        self.current_input = current_input
        self.suggestions = text_suggestion.suggest_text(
            text=' '.join(tokenize(current_input)),
            n_words=2,
            n_texts=4,
            beam_width=3
        )

    def update_current_input(self, new_current_input: str):
        """
        :param: **new_current_input** - текст, на который пользователь кликнул.
        """
        self.current_input = new_current_input
        self.update_suggestions(self.current_input)


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.input(
                placeholder="Start writing here",
                value=State.current_input,
                on_change=lambda value: State.update_suggestions(value),
                style={
                    "width": "1000px",
                    "height": "50px",
                    "font-size": "18px"
                }
            ),
            rx.list(
                rx.foreach(
                    State.suggestions,
                    lambda x: rx.list.item(
                        rx.button(
                            x,
                            on_click=State.update_current_input(x)
                        )
                    )
                ),
            ),
        ),
        width="100%",
        height="100vh",
    )


app = rx.App()
app.add_page(index)
