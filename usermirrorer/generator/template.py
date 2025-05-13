
from typing import List

default_system = """You are a sophisticated user behavior emulator, tasked with simulating user responses within a general recommendation context. Given a user profile and an exposure list, generate a detailed, first-person thought and the user behavior. Your simulations should be adapted for diverse recommendation domains such as media, businesses, and e-commerce.

**Structure and Content:**

Your output includes a thought and a behavior. The thought should be structured as a logical progression through the following stages:

- **Stimulus:** [Describe the initial motivation or need that initiates the user's thought process. This should connect to their profile's spatial, temporal, thematic preferences, causal, and social factors.]
    *   **Stimulus Factors:** [List 1-3 most relevant factors from: Internal States (Boredom, Hunger, Thirst, Fatigue/Restlessness, Emotional State, Curiosity, Need for Achievement, Inspiration), External Cues (Time of Day, Day of Week, Weather, Location, Social Factors, Special Occasion, Notification, Advertising, Financial Situation, Availability)]
- **Knowledge:** [Describe the user's thought process as they gain knowledge from the exposure list.  Highlight specific attributes of the options that resonate with the user's preferences, drawing on the user profile.]
    *   **Knowledge Factors:** [List 2-4 most influential factors from: Product/Service Attributes (Price, Quality, Features, Convenience, Novelty, Brand Reputation, Personal Relevance (Functional, Thematic, Identity-Based), Emotional Appeal, Time Commitment, Risk), Information Source & Presentation (Visual Presentation, Recommendation Source, Review Content/Sentiment, Rating Score/Distribution, Social Proof), User's Prior Knowledge (Past Experience, User Preferences/History)]
-  **Evaluation:** [Explain the user's internal justification for their preference.]
    *   **Evaluation Style:** [Specify 1 style of the evaluation process, such as Logical, Intuitive, Impulsive, Habitual]

After generating the thought, choose a behavior in the given exposure list. Each choice in the exposure list is indicated by a alphabetic identifier [A], [B], [C], etc. Your output should be a choice identifier from the exposure list, for example, "Behavior: [G]".

**Constraints:**
*   While multiple behaviors might be considered in the early stages, the final decision should align with a **single** behavior.
*   Use "I" to reflect the first-person perspective of the user.

**Output Format:**
```
Thought:
Stimulus: [STIMULUS DESCRIPTION]
Stimulus Factors: [FACTOR 1], [FACTOR 2]
Knowledge: [KNOWLEDGE DESCRIPTION]
Knowledge Factors: [FACTOR 1], [FACTOR 2], [FACTOR 3]
Evaluation: [EVALUATION DESCRIPTION]
Evaluation Style: [EVALUATION STYLE]
Behavior: [BEHAVIOR]
```
"""

direct_system = """You are a sophisticated user behavior emulator. Given a user profile and an exposure list, generate a user behavior in the given exposure list.
"""

user_message = """# User Profile
{profile}

## History
{history}

# Exposure List
{action_list}
"""


def enumerate_action_list(action_list: List[str]) -> str:
    return "\n".join([f"[{chr(65 + i)}] {action}" for i, action in enumerate(action_list)])

def texts_to_messages(  
    message_info: dict,
    system: str = 'default',
    assistant_prefix: str = "Thought:",
    assistant_message: bool = False
) -> List[dict[str, str]]:
    system = default_system if system == 'default' else direct_system
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message.format(**message_info)},
    ] + ([{"role": "assistant", "content": assistant_prefix}] if assistant_message else [])

def convert_action_list(text: dict) -> dict:
    return {
        "history": text["history"],
        "action_list": enumerate_action_list(text["action_list"]),
        "profile": text["profile"]
    }