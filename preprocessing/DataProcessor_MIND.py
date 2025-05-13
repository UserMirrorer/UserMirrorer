import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, default="")
parser.add_argument("--project_path", type=str, default="")
args = parser.parse_args()

DATASET_NAME = "MIND"
DATASET_PATH = args.source_path
OUTPUT_PATH = args.project_path + "/raws/"
MIN_INTERACTION_CNT = 5     # The minimum number of interactions for a user to be included in the dataset.
MAX_INTERACTION_CNT = 20  # The maximum number of interactions for a user to be included in the dataset.

import os
# List all files in the dataset directory
print("Files in dataset directory:")
for file in os.listdir(DATASET_PATH):
    print(f"- {file}")

# Step 1.2 Preprocess
import pandas as pd

impression = pd.read_csv(os.path.join(DATASET_PATH, "behaviors_valid.tsv"), sep="\t", header=None, names=["impression_id", "user_id", "time", "history", "impressions"])
impression.to_csv(os.path.join(DATASET_PATH, "behaviors.tsv"), index=False, sep="\t", header=None, mode="a")

df = pd.read_csv(os.path.join(DATASET_PATH, "news.tsv"), sep="\t", header=None, names=["news_id", "category", "sub_category", "title", "abstract", "url", "title_entities", "abstract_entities"])
df_2 = pd.read_csv(os.path.join(DATASET_PATH, "news_valid.tsv"), sep="\t", header=None, names=["news_id", "category", "sub_category", "title", "abstract", "url", "title_entities", "abstract_entities"])
df_3 = pd.read_csv(os.path.join(DATASET_PATH, "news_test.tsv"), sep="\t", header=None, names=["news_id", "category", "sub_category", "title", "abstract", "url", "title_entities", "abstract_entities"])
#  # Combine the 3 dataframes
news = pd.concat([df, df_2, df_3], ignore_index=True, axis=0)

# # Display the first 5 rows
news.drop_duplicates("news_id", inplace=True)

news.to_csv(os.path.join(DATASET_PATH, "news_all.tsv"), index=False, sep="\t", header=None)

from datetime import datetime
from tqdm import tqdm
tqdm.pandas()


def behavior_preprocess(path, min_history_length=5, min_behavior_counts=5, max_impressions_length=10):

    behavior_chunks = pd.read_csv(os.path.join(path, "behaviors.tsv"), sep="\t", header=None, names=["impression_id", "user_id", "time", "history", "impressions"], chunksize=100000)
    new_behaviors = []
    for behaviors in behavior_chunks:
      behaviors['history'] = behaviors['history'].progress_apply(lambda x: x.split() if pd.notna(x) else [])
      behaviors = behaviors[behaviors['history'].apply(len) >= min_history_length]

      behaviors['impressions'] = behaviors['impressions'].apply(lambda x: x.split())

      behaviors = behaviors[behaviors['impressions'].apply(len) <= max_impressions_length]

      behaviors = behaviors[behaviors['impressions'].apply(lambda x: sum([imp.endswith("1") for imp in x]) == 1)]
      
      new_behaviors.append(behaviors)
    return pd.concat(new_behaviors, axis=0)




def load_news(path: str, filename: str = "news_all.tsv") -> pd.DataFrame:
    news = pd.read_csv(
        os.path.join(path, filename),
        sep="\t", header=None,
        names=[
            "news_id", "category", "sub_category",
            "title", "abstract", "url",
            "title_entities", "abstract_entities"
        ]
    )
    return news

def process_impressions(impressions):
    item_ids = []
    positions = []
    
    for idx, imp in enumerate(impressions):
        item_id, value = imp.split('-')
        item_ids.append(item_id)
        if value == '1':
            positions.append(idx)
    return item_ids, positions

def process_user_impressions(input_df):
    processed_impressions = input_df['impressions'].apply(process_impressions)
    
    input_df['item_list'] = processed_impressions.apply(lambda x: x[0])
    input_df['label'] = processed_impressions.apply(lambda x: x[1])
    
    return input_df

behaviors = behavior_preprocess(DATASET_PATH)

impression = process_user_impressions(behaviors)

news = load_news(DATASET_PATH)
times1 = impression.loc[:, ['time', 'item_list']].explode('item_list').groupby('item_list')['time'].agg('min')
times2 = impression.loc[:, ['time', 'history']].explode('history').groupby('history')['time'].agg('min')
item_list = pd.concat([times1, times2]).groupby(level=0).min().to_frame()
news = news.join(item_list, on='news_id')
# %%
# Step 4: Convert the dataset to the unified format

from tqdm import tqdm
from datetime import datetime

tqdm.pandas()

item_df = news.progress_apply(
        lambda row: {
            "item_id": row["news_id"],
            "item_description": {
                "title": row["title"],
                "abstract": row["abstract"] if row["abstract"] else "NA",
                "category": row["category"],
                "subcategory": row["sub_category"]
            },
            "item_features": {
                "title_entities": row["title_entities"] if pd.notna(row["title_entities"]) else "",
                "abstract_entities": row["abstract_entities"] if pd.notna(row["abstract_entities"]) else "",
                "time": datetime.strptime(row["time"], '%m/%d/%Y %H:%M:%S %p').timestamp() if pd.notna(row["time"]) else 0
            }
        },
        axis=1
    ).tolist()

user_df = impression.groupby('user_id').agg({
    'history': 'first'  # Take the first history for each user
}).progress_apply(
    lambda row: {
        "user_id": row.name,  # row.name contains the user_id in grouped data
        "user_description": {},  # Empty since we don't have user descriptions
        "user_features": {
            "history": row["history"]
        }
    },
    axis=1
).tolist()

interaction_df = impression.progress_apply(
    lambda row: {
        "user_id": row["user_id"],
        "item_id": row["item_list"][row["label"][0]],
        "timestamp": row["time"],
        "behavior_features": {
            "impression_list": row["item_list"],
            "item_pos": row["label"][0]
        }
    },
    axis=1
).tolist()


item_df = pd.DataFrame(item_df)
user_df = pd.DataFrame(user_df)
interaction_df = pd.DataFrame(interaction_df)


print(f"\nFinal shapes:")
print(f"Items: {len(item_df)}")
print(f"Users: {len(user_df)}")
print(f"Interactions: {len(interaction_df)}")


print("\nSaving files...")
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
interaction_df.to_json(os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_interaction.jsonl"), lines=True, orient="records")
user_df.to_json(os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_user_feature.jsonl"), lines=True, orient="records")
item_df.to_json(os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_item_feature.jsonl"), lines=True, orient="records")

print("\nProcessing complete.")
