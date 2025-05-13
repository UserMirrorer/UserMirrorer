import pandas as pd
from transformers import AutoTokenizer

from ..generator.template import texts_to_messages, convert_action_list

def length_filtering(
    data: pd.DataFrame,
    tokenizer_path: str,
    length: int,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = data.copy()
    df['messages'] = df['text'].apply(lambda x: texts_to_messages(convert_action_list(x)))
    df['token_length'] = tokenizer.apply_chat_template(df['messages'].tolist(), tokenize=True)
    df['token_length'] = df['token_length'].apply(len)
    df = df[df['token_length'] <= length]
    df = df.drop(columns=['token_length', 'messages'])

    print(f"Filtered {len(df)} samples out of {len(data)}")
    return df