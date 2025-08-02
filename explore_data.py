import pandas as pd
import numpy as np

print("Loading raw data...")
df = pd.read_csv('train.csv')

conditions = [
    df['winner_model_a'] == 1,
    df['winner_model_b'] == 1,
    df['winner_tie'] == 1
]
choices = [0, 1, 2]
df['winner'] = np.select(conditions, choices, default=-1) # default should not be used if data is clean

sep_token = "[SEP]"
df['input_text'] = df['prompt'] + sep_token + df['response_a'] + sep_token + df['response_b']

processed_df = df[['input_text', 'winner']].copy()

processed_df.to_csv('processed_train.csv', index=False)

print("Processed data saved to 'processed_train.csv'")
print("\n--- First 5 Rows of Processed Data ---")

pd.set_option('display.max_colwidth', 200)
print(processed_df.head())