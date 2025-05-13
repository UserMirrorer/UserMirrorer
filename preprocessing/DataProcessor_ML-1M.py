
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, default="")
parser.add_argument("--project_path", type=str, default="")
args = parser.parse_args()  

DATASET_NAME = "ml-1m"
DATASET_PATH = args.source_path
OUTPUT_PATH = args.project_path + "/raws/"
MIN_INTERACTION_CNT = 5     # The minimum number of interactions for a user to be included in the dataset.
MAX_INTERACTION_CNT = 20  # The maximum number of interactions for a user to be included in the dataset.

import os
# List all files in the dataset directory
print("Files in dataset directory:")
for file in os.listdir(DATASET_PATH):
    print(f"- {file}")

import pandas as pd
# Load the ratings data
ratings_df = pd.read_csv(os.path.join(DATASET_PATH, "ratings.dat"), 
                        sep="::", 
                        header=None, 
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                        engine='python')

# Load the users data
users_df = pd.read_csv(os.path.join(DATASET_PATH, "users.dat"),
                      sep="::",
                      header=None,
                      names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                      engine='python')

# Load the movies data
movies_df = pd.read_csv(os.path.join(DATASET_PATH, "movies.dat"),
                       sep="::",
                       header=None,
                       names=['MovieID', 'Title', 'Genres'],
                       encoding='latin-1',
                       engine='python')

# Display the first few rows of each dataframe
print("Ratings DataFrame:")
print(ratings_df.head())
print("\nUsers DataFrame:")
print(users_df.head())
print("\nMovies DataFrame:")
print(movies_df.head())


ITEM_DESCRIPTION_TEMPLATE = "'{title}' [{genres}]"

def create_movie_description(row):
    # Extract year from title if present
    title = row['Title']
    
    # Convert genres from '|' separated string to list
    genres = row['Genres']

    # Create description using a single format string
    description = ITEM_DESCRIPTION_TEMPLATE.format(title=title, genres=genres)
    
    return description

# Create text descriptions for movies
movies_df['Description'] = movies_df.apply(create_movie_description, axis=1)

print("Example movie descriptions:")
print(movies_df[['MovieID', 'Description']].head())


Gender_map = {
    "F": "female",
    "M": "male"
}

Age_map = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45+"
}

Occupation_map = {
    0: "other",
    1: "academic/educator", 
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}
    
from uszipcode import SearchEngine

def get_zipcode_info(zipcode):
    try:
        search = SearchEngine()
        result = search.by_zipcode(str(zipcode))
        if result:
           return result.city + ", " + result.state
        return None
    except:
        return None

Zipcode_map = {}
unique_zipcodes = users_df['Zip-code'].unique()

for zipcode in unique_zipcodes:
    if pd.notna(zipcode):  # Skip NaN values
        info = get_zipcode_info(zipcode)
        if info:
            Zipcode_map[zipcode] = info

users_df['Zip-code'] = users_df['Zip-code'].apply(lambda x: Zipcode_map.get(x, "Unknown"))
users_df['Gender'] = users_df['Gender'].apply(lambda x: Gender_map.get(x, "Unknown"))
users_df['Age'] = users_df['Age'].apply(lambda x: Age_map.get(x, "Unknown"))
users_df['Occupation'] = users_df['Occupation'].apply(lambda x: Occupation_map.get(x, "Unknown"))
USER_PROFILE_TEMPLATE = """Gender: {gender}, Age: {age}, Occupation: {occupation}, Location: {zipcode}
"""

users_df['Profile'] = users_df.apply(lambda row: USER_PROFILE_TEMPLATE.format(user_id=row['UserID'], gender=row['Gender'], age=row['Age'], occupation=row['Occupation'], zipcode=row['Zip-code']), axis=1)

user_df = users_df
user_df = users_df.rename(columns={'UserID': 'user_id', 'Gender': 'gender', 'Age': 'age', 'Occupation': 'occupation', 'Zip-code': 'zipcode'})
item_df = movies_df
item_df = item_df.rename(columns={'MovieID': 'item_id', 'Title': 'title', 'Genres': 'genres'})
interactions_df = ratings_df
interactions_df = interactions_df.rename(columns={'UserID': 'user_id', 'MovieID': 'item_id', 'Rating': 'rating', 'Timestamp': 'timestamp'})

user_count = interactions_df.groupby('user_id').size()
user_count = user_count[user_count >= MIN_INTERACTION_CNT]

interactions_df = interactions_df[interactions_df['user_id'].isin(user_count.index)]

import json
import re
from datetime import datetime
from tqdm import tqdm
import numpy as np

def extract_year_from_title(title):
    """Extract year from title if present"""
    # Regular expression to find year in parentheses
    year_match = re.search(r'\((\d{4})\)', title)
    if year_match:
        return year_match.group(1)
    return None

def parse_date(date_str):
    """Convert various date formats to YYYY-MM-DD HH:MM:SS format"""
    if isinstance(date_str, (int, np.int64)):
        # If input is timestamp (integer), convert it to datetime
        try:
            dt = datetime.fromtimestamp(date_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None
            
    if not date_str or (isinstance(date_str, str) and date_str.strip() == ''):
        return None
        
    try:
        # 尝试解析Goodreads格式的日期
        dt = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None


def process_movie(movie):
    
    item_description = {
        "title": movie['title'],
        "genres": movie['genres'],
        "year": extract_year_from_title(movie['title'])
    }
    
    item_features = {
        "year": extract_year_from_title(movie['title'])
    }
    
    return {
        "item_id": str(movie['item_id']),
        "item_description": item_description,
        "item_features": item_features
    }

def process_user(user):
    user_description = {
        "gender": user['gender'],
        "age": user['age'],
        "occupation": user['occupation'],
        "location": user['zipcode']
    }
    
    user_features = {
    }
    return {
        "user_id": str(user['user_id']),
        "user_description": user_description,
        "user_features": user_features
    }

def process_interaction(interaction):
    behavior_features = {
        "rating": float(interaction['rating']) if pd.notna(interaction['rating']) else "Not Rated"
    }
    
    timestamp = parse_date(interaction['timestamp'])
    if not timestamp:
        return None
        
    return {
        "user_id": str(interaction['user_id']),
        "item_id": str(interaction['item_id']),
        "timestamp": timestamp,
        "behavior_features": behavior_features
    }

# Process items (books)
tqdm.pandas(desc="Processing books")
item_records = item_df.progress_apply(process_movie, axis=1).tolist()

# Process users 
tqdm.pandas(desc="Processing users")
user_records = user_df.progress_apply(process_user, axis=1).tolist()

# Process interactions
tqdm.pandas(desc="Processing interactions")
interaction_records = interactions_df.progress_apply(process_interaction, axis=1).dropna().tolist()

import os
# Convert records to pandas DataFrames
interaction_df = pd.DataFrame.from_records(interaction_records)
user_df = pd.DataFrame.from_records(user_records) 
item_df = pd.DataFrame.from_records(item_records)

os.makedirs(OUTPUT_PATH, exist_ok=True)
interaction_df.to_json(os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_interaction.jsonl"), lines=True, orient="records")
user_df.to_json(os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_user_feature.jsonl"), lines=True, orient="records")
item_df.to_json(os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_item_feature.jsonl"), lines=True, orient="records")
