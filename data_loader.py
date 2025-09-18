import pandas as pd
import numpy as np

def load_story_points_dataset(file_path="sample_stories.csv"):
    """Load 100 user stories from CSV"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Basic preprocessing for story point data"""
    df = df.dropna()
    df['description'] = df['description'].str.strip()
    return df