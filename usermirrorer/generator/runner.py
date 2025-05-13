"""
This module implements a runner for batch processing text data through a language model classifier.
It handles loading data, processing it through an LLM, and saving the results.
The main functionality includes converting text inputs into a specific message format and
generating classifications based on multiple-choice options.
"""

import pandas as pd
from tqdm import tqdm

from .classifier import LLMClassifier
from .template import texts_to_messages


def run(df: pd.DataFrame, model: LLMClassifier, choice_cnt: int, repeat_times: int = 1) -> pd.DataFrame:
    """
    Process a subset of the dataframe with specific number of choices through the LLM classifier.
    
    Args:
        df (pd.DataFrame): Input dataframe containing messages to process
        model (LLMClassifier): The LLM classifier instance to use
        choice_cnt (int): Number of choices for classification
        repeat_times (int, optional): Number of times to repeat each classification
        
    Returns:
        pd.DataFrame: Updated dataframe with classification results
    """
    # Filter dataframe for rows with specified choice count
    df_slice = df[df["choice_cnt"] == choice_cnt].copy()
    message_with_decision = df_slice['messages']
    
    # Run the model on the filtered messages
    output, messages = model.run_multiturn(
        message_with_decision.tolist(),
        choices=[f"{chr(65 + i)}" for i in range(choice_cnt)],  # Generate choices as A, B, C, etc.
        stop_turn=["Behavior: ["],
        repeat_times=repeat_times
    )
    
    # Update the original dataframe with results
    df.loc[df_slice.index, "choice"] = pd.Series(output, index=df_slice.index)
    df.loc[df_slice.index, "decision_list"] = pd.Series(messages, index=df_slice.index)
    return df

def run_all(df: pd.DataFrame, model: LLMClassifier, repeat_times: int = 1):
    """
    Process the entire dataframe through the LLM classifier, handling different choice counts.
    
    Args:
        df (pd.DataFrame): Input dataframe containing all messages
        model (LLMClassifier): The LLM classifier instance to use
        repeat_times (int, optional): Number of times to repeat each classification
        
    Returns:
        pd.DataFrame: Complete dataframe with all classification results
    """
    # Convert text to formatted messages with decision and stimulus structure
    df['messages'] = df['text'].apply(lambda x: texts_to_messages(x, assistant_message=True, assistant_prefix="Thought:\nStimulus:"))
    df["choice"] = None
    df["decision_list"] = None
    for choice_cnt in tqdm(df["choice_cnt"].unique().tolist()):
        # Skip already processed choice counts
        if df[df["choice_cnt"] == choice_cnt]["choice"].notna().all():
            print(f"Choice {choice_cnt} already generated, skip")
            continue
        df = run(df, model, choice_cnt, repeat_times)
    return df