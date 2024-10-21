import pandas as pd
import numpy as np
import re
import email
from IPython.display import clear_output
from collections import defaultdict
from typing import Union, List

def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        body = e.get_payload()
        pattern = r"^(From:|To:|Sent:|Subject:|Importance:|\[mailto:| -|----|\t|When:|Date:|Time:|Where:|href=|Location:|Cc:|Video:|Conf call:|Start Date:|cc:|Phone:|Facsimile:|E-Mail:|Sender:|Attachment Type:|>|<|\*{5}).*"
        cleaned_body = re.sub(pattern, '', body, flags=re.MULTILINE)
        cleaned_body = re.sub(r'[^A-Za-z\n ,.]', '', cleaned_body)
        cleaned_body = re.sub(r' +', ' ', cleaned_body).strip()
        cleaned_body = re.sub(r'\t', ' ', cleaned_body).strip()
        cleaned_body = re.sub(r'\n+', '\n', cleaned_body).strip()
        column.append(cleaned_body)
    return column


def tokenize(text):
    reg = re.compile(r'\w+|,|\.|!|\?')
    return reg.findall(text.lower())

