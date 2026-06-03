import pandas as pd

df = pd.read_csv('processed/combined_data.csv', dtype={'Beep': 'str'})

print(df[df["Age_group"] == "U17"].sort_values("Q_birth", ascending=False))