import pandas as pd

# Charge le fichier
df = pd.read_parquet("data/features/segments_mfcc.parquet")

# Affiche les 5 premières lignes et les colonnes
print(df.info())
print(df.head())

print(f"Nombre total de segments : {len(df)}")