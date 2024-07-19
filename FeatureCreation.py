import pandas as pd
from DirectedEvolution.FeatureCreationFunction import *
from collections import Counter
from itertools import combinations, product
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
df = pd.read_csv(r'C:\Users\shash\Downloads\ChatBoxPython\CodingProjects\XylaneFinalData\trainingdata2')
de = DirectedEvolution()
print(df.head)
df.rename(columns={'Mutated Sequence': 'P_seq'}, inplace=True)
print(df.columns)
def remove_invalid_sequences(df, column_name):
    valid_rows = []
    for index, row in df.iterrows():
        sequence = row[column_name]
        invalid_chars = set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')
        if not invalid_chars:
            valid_rows.append(index)
    return df.loc[valid_rows]
df = remove_invalid_sequences(df, 'P_seq')
# Assuming df['P_seq'] contains the protein sequences
df['Protein_Length'] = df['P_seq'].apply(lambda sequence: de.calculate_protein_length(sequence))
df['Isoelectric_Point'] = df['P_seq'].apply(de.calculate_isoelectric_point)
df['Molecular_Weight'] = df['P_seq'].apply(de.calculate_molecular_weight)
df['Aromaticity'] = df['P_seq'].apply(de.calculate_aromaticity)
df['Instability_Index'] = df['P_seq'].apply(de.calculate_instability_index)
df['Gravy'] = df['P_seq'].apply(de.calculate_gravy)
df['Helix'], df['Turn'], df['Sheet'] = zip(*df['P_seq'].apply(de.calculate_secondary_structure_fraction))

# Optionally, drop the 'Sequence' column if you no longer need it
df.drop(['P_seq', 'OGT'], axis=1, inplace=True)

# Display the updated DataFrame
print(df.shape)

# Save the final DataFrame to a CSV file
df.to_csv(r'C:\Users\shash\Downloads\ChatBoxPython\CodingProjects\XylaneFinalData\trainingdata3', index=False)
