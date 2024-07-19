from multiprocessing import Pool
from functools import partial
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm
import joblib
import os
from DirectedEvolution.FeatureCreationFunction import DirectedEvolution

def process_sequence(i, p_seq, dt_model):
    de = DirectedEvolution()

    # Generate mutations for a given protein sequence
    protein_sequence = p_seq
    mutations_df = de.generate_mutations(protein_sequence)
    mutations_df.rename(columns={'P_seq': 'Mutated Sequence'}, inplace=True)
    
    def remove_invalid_sequences(df, column_name):
        valid_rows = []
        for index, row in df.iterrows():
            sequence = row[column_name]
            invalid_chars = set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')
            if not invalid_chars:
                valid_rows.append(index)
        return df.loc[valid_rows]
    mutations_df = remove_invalid_sequences(mutations_df, 'Mutated Sequence')
    
    print("MUTATION CREATION DONE")
    monomers = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    dipeptides = [aa1 + aa2 for aa1 in monomers for aa2 in monomers]
    print('STARTING MONOMER')
    for monomer in monomers:
        mutations_df[f'{monomer}_frequency'] = mutations_df['Mutated Sequence'].apply(
            lambda sequence: de.calculate_monomer_frequency(sequence=sequence).get(monomer, 0)
        )
    print('STARTING DIPEPTIDE')
    for dipeptide in dipeptides:
        mutations_df[dipeptide] = mutations_df['Mutated Sequence'].apply(
            lambda sequence: de.calculate_dipeptide_frequency(sequence).get(dipeptide, 0)
        )
    print('STARTING CALCULATIONS')
    with Pool() as pool:
        calculate_properties_partial = partial(de.calculate_properties, 
                                               steric_parameter=True,
                                               polarizability=True,
                                               volume=True,
                                               hydrophobicity=True,
                                               isoelectric_point=True,
                                               helix_prob=True,
                                               sheet_prob=True)
        properties_results = pool.map(calculate_properties_partial, mutations_df['Mutated Sequence'])
        
    mutations_df['Protein_Length'] = mutations_df['Mutated Sequence'].apply(lambda sequence: de.calculate_protein_length(sequence))
    mutations_df['Isoelectric_Point'] = mutations_df['Mutated Sequence'].apply(de.calculate_isoelectric_point)
    mutations_df['Molecular_Weight'] = mutations_df['Mutated Sequence'].apply(de.calculate_molecular_weight)
    mutations_df['Aromaticity'] = mutations_df['Mutated Sequence'].apply(de.calculate_aromaticity)
    mutations_df['Instability_Index'] = mutations_df['Mutated Sequence'].apply(de.calculate_instability_index)
    mutations_df['Gravy'] = mutations_df['Mutated Sequence'].apply(de.calculate_gravy)
    mutations_df['Helix'], mutations_df['Turn'], mutations_df['Sheet'] = zip(*mutations_df['Mutated Sequence'].apply(de.calculate_secondary_structure_fraction))
    
    df_properties = pd.DataFrame(properties_results)
    df = pd.concat([mutations_df, df_properties], axis=1)
    print('FEATURES CREATED')

    if 'turn_prob' in df.columns:
        df = df.drop(['turn_prob'], axis=1)
    df.rename(columns={'Mutated Sequence': 'P_seq'}, inplace=True)

    # Set the OGT column to 36°C for Aspergillus niger
    df['OGT'] = 36
    
    output_filename = f'iteration_{i}_mutations.csv'
    df.to_csv(output_filename, index=False)
    print(f'Results saved to {output_filename}')
    
    mutations = df['Mutation']
    m_pseq = df['P_seq']
    df = df.drop(['Mutation', 'P_seq'], axis=1).dropna()
    print(df.columns)
    print(df.head)
    scaler = StandardScaler()
    features = scaler.fit_transform(df)
    pred = dt_model.predict(features)
    
    topt = pd.DataFrame(pred, columns=['Topt'])
    
    final_df = pd.concat([df, topt], axis=1)
    final_df = pd.concat([final_df, mutations, m_pseq], axis=1)
    columns_to_save = ['OGT', 'Topt', 'Mutation', 'P_seq']
    final_df = final_df[columns_to_save]

    # Save the final DataFrame to a CSV file
    output_filename = f'iteration_{i}_results.csv'
    final_df.to_csv(output_filename, index=False)
    print(f'Results saved to {output_filename}')

def apply_mutations(sequence, mutations):
    """
    Apply mutations to an amino acid sequence.

    Args:
    sequence (str): The original amino acid sequence.
    mutations (str): The mutations in the format "G200K_G140S".

    Returns:
    str: The mutated amino acid sequence.
    """
    # Split the mutations by '_'
    mutation_list = mutations.split('_')

    # Convert sequence to a list for easier manipulation
    sequence_list = list(sequence)

    # Apply each mutation
    for mutation in mutation_list:
        # Parse the mutation
        original_aa = mutation[0]            # Original amino acid
        position = int(mutation[1:-1]) - 1   # Position (convert to 0-based index)
        new_aa = mutation[-1]                # New amino acid

        # Check if the original amino acid at the specified position matches
        if sequence_list[position] != original_aa:
            raise ValueError(f"Expected {original_aa} at position {position + 1}, but found {sequence_list[position]}")

        # Apply the mutation
        sequence_list[position] = new_aa

    # Convert the list back to a string
    mutated_sequence = ''.join(sequence_list)
    return mutated_sequence

if __name__ == "__main__":
    joblib_file = "best_decision_tree_model.pkl"
    dt_model = joblib.load(joblib_file)

    for i in tqdm(range(16), desc="Processing Iterations"):
        print(i)
        if i == 0:
            protein_sequence = "MLTKNLLLCFAAAKAALAVPHDSVAQRSDALHMLSERSTPSSTGENNGFYYSFWTDGGGDVTYTNGDAGAYTVEWSNVGNFVGGKGWNPGSAQDITYSGTFTPSGNGYLSVYGWTTDPLIEYYIVESYGDYNPGSGGTYKGTVTSDGSVYDIYTATRTNAASIQGTATFTQYWSVRQNKRVGGTVTTSNHFNAWAKLGMNLGTHNYQIVATEGYQSSGSSSITVQ"
        else:
            iteration = i - 1
            df = pd.read_csv(f'iteration_{iteration}_results.csv', delimiter=',')
            df.sort_values('Topt', ascending=False, inplace=True)
            print(df.head(1))
            mutation = df['Mutation'].iloc[0]
            protein_sequence = apply_mutations(protein_sequence, mutation)

        process_sequence(i, protein_sequence, dt_model)
