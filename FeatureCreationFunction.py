import pandas as pd
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


class DirectedEvolution:
    def generate_mutations(self, protein_sequence, max_mutations=27000):
        amino_acids = 'ACDEFGHIKLNPQRSTVWYM'  # Standard 20 amino acids
        mutations_set = set()
        sequence_length = len(protein_sequence)
        '''
        # Generate all possible single mutations
        for position in range(sequence_length):
            original_aa = protein_sequence[position]
            for new_aa in amino_acids:
                if new_aa != original_aa:
                    mutated_sequence = list(protein_sequence)
                    mutated_sequence[position] = new_aa
                    mutation_description = f"{original_aa}{position+1}{new_aa}"
                    mutation = ''.join(mutated_sequence)
                    mutations_set.add((mutation, mutation_description))
        '''
        # Generate random double mutations until we have enough mutations
        while len(mutations_set) < max_mutations:
            position1, position2 = random.sample(range(sequence_length), 2)
            original_aa1 = protein_sequence[position1]
            original_aa2 = protein_sequence[position2]
            new_aa1 = random.choice([aa for aa in amino_acids if aa != original_aa1])
            new_aa2 = random.choice([aa for aa in amino_acids if aa != original_aa2])

            mutated_sequence = list(protein_sequence)
            mutated_sequence[position1] = new_aa1
            mutated_sequence[position2] = new_aa2
            mutation_description = f"{original_aa1}{position1+1}{new_aa1}_{original_aa2}{position2+1}{new_aa2}"
            mutation = ''.join(mutated_sequence)

            mutations_set.add((mutation, mutation_description))

        mutations_df = pd.DataFrame(list(mutations_set), columns=['P_seq', 'Mutation'])
        return mutations_df
    
    def calculate_monomer_frequency(self, sequence):
        monomer_counts = Counter(sequence)
        total_monomers = sum(monomer_counts.values())
        monomer_frequencies = {monomer: count / total_monomers for monomer, count in monomer_counts.items()}
        return monomer_frequencies
    
    def calculate_dipeptide_frequency(self, sequence):
        if not isinstance(sequence, (str, list)):
            print(f"Error: sequence should be a string or a list, but got {type(sequence)}")
            return None
        dipeptide_counts = Counter([sequence[i:i + 2] for i in range(len(sequence) - 1)])
        total_dipeptides = sum(dipeptide_counts.values())
        dipeptide_frequencies = {dipeptide: count / total_dipeptides for dipeptide, count in dipeptide_counts.items()}
        return dipeptide_frequencies
    
    def calculate_steric_parameter(self, sequence):
        StericParameters = {
            'A': 1.28, 'G': 0.0, 'V': 3.67, 'L': 2.59, 'I': 4.19, 'F': 2.94, 'Y': 2.94,
            'W': 3.21, 'T': 3.03, 'S': 1.31, 'R': 2.34, 'K': 1.89, 'H': 2.99, 'D': 1.6,
            'E': 1.56, 'N': 1.6, 'Q': 1.56, 'M': 2.35, 'P': 2.67, 'C': 1.77
        }
        VS = 0
        for aa in sequence:
            VS += StericParameters.get(aa, 2.4)
        average_steric_parameter = VS / len(sequence)
        return average_steric_parameter
    
    def calculate_polarizability(self, sequence):
        Polarizability = {
            'A': 0.05, 'G': 0.0, 'V': 0.14, 'L': 0.19, 'I': 0.19, 'F': 0.29, 'Y': 0.3,
            'W': 0.41, 'T': 0.11, 'S': 0.06, 'R': 0.29, 'K': 0.22, 'H': 0.23, 'D': 0.11,
            'E': 0.15, 'N': 0.13, 'Q': 0.18, 'M': 0.22, 'P': 0.0, 'C': 0.13
        }
        VS = 0
        for aa in sequence:
            VS += Polarizability.get(aa, 2.4)
        average_polarizability = VS / len(sequence)
        return average_polarizability
    
    def calculate_volume(self, sequence):
        Volume = {
            'A': 1.0, 'G': 0.0, 'V': 3.0, 'L': 4.0, 'I': 4.0, 'F': 5.89, 'Y': 6.47,
            'W': 8.08, 'T': 2.6, 'S': 1.6, 'R': 6.13, 'K': 4.77, 'H': 4.66, 'D': 2.78,
            'E': 3.78, 'N': 2.95, 'Q': 3.95, 'M': 4.43, 'P': 2.72, 'C': 2.43
        }
        VS = 0
        for aa in sequence:
            VS += Volume.get(aa, 2.4)
        average_volume = VS / len(sequence)
        return average_volume
    
    def calculate_hydrophobicity(self, sequence):
        Hydrophobicity = {
            'A': 0.31, 'G': 0.0, 'V': 1.22, 'L': 1.7, 'I': 1.8, 'F': 1.79, 'Y': 0.96,
            'W': 2.25, 'T': 0.26, 'S': -0.04, 'R': -1.01, 'K': -0.99, 'H': 0.13, 'D': -0.77,
            'E': -0.64, 'N': -0.6, 'Q': -0.22, 'M': 1.23, 'P': 0.72, 'C': 1.54
        }
        VS = 0
        for aa in sequence:
            VS += Hydrophobicity.get(aa, 2.4)
        average_hydrophobicity = VS / len(sequence)
        return average_hydrophobicity
    
    def calculate_isoelectric_point(self, sequence):
        Isoelectric_point = {
            'A': 6.11, 'G': 6.07, 'V': 6.02, 'L': 6.04, 'I': 6.04, 'F': 5.67, 'Y': 5.66,
            'W': 5.94, 'T': 5.6, 'S': 5.7, 'R': 10.74, 'K': 9.99, 'H': 7.69, 'D': 2.95,
            'E': 3.09, 'N': 6.52, 'Q': 5.65, 'M': 5.71, 'P': 6.8, 'C': 6.35
        }
        VS = 0
        for aa in sequence:
            VS += Isoelectric_point.get(aa, 2.4)
        average_isoelectric_point = VS / len(sequence)
        return average_isoelectric_point
    
    def calculate_helix_prob(self, sequence):
        Helix_prob = {
            'A': 0.42, 'G': 0.13, 'V': 0.27, 'L': 0.39, 'I': 0.3, 'F': 0.3, 'Y': 0.25,
            'W': 0.32, 'T': 0.21, 'S': 0.2, 'R': 0.36, 'K': 0.32, 'H': 0.27, 'D': 0.25,
            'E': 0.42, 'N': 0.21, 'Q': 0.36, 'M': 0.38, 'P': 0.13, 'C': 0.17
        }
        VS = 0
        for aa in sequence:
            VS += Helix_prob.get(aa, 2.4)
        average_helix_prob = VS / len(sequence)
        return average_helix_prob
    
    def calculate_sheet_prob(self, sequence):
        Sheet_prob = {
            'A': 0.23, 'G': 0.15, 'V': 0.49, 'L': 0.31, 'I': 0.45, 'F': 0.38, 'Y': 0.41,
            'W': 0.42, 'T': 0.36, 'S': 0.28, 'R': 0.25, 'K': 0.27, 'H': 0.3, 'D': 0.2,
            'E': 0.21, 'N': 0.22, 'Q': 0.25, 'M': 0.32, 'P': 0.34, 'C': 0.41
        }
        VS = 0
        for aa in sequence:
            VS += Sheet_prob.get(aa, 2.4)
        average_sheet_prob = VS / len(sequence)
        return average_sheet_prob
        
    def calculate_protein_length(self, sequence):
        return len(sequence)

    def calculate_isoelectric_point(self, sequence):
        protein = ProteinAnalysis(sequence)
        return protein.isoelectric_point()

    def calculate_molecular_weight(self, sequence):
        protein = ProteinAnalysis(sequence)
        return protein.molecular_weight()

    def calculate_aromaticity(self, sequence):
        protein = ProteinAnalysis(sequence)
        return protein.aromaticity()

    def calculate_instability_index(self, sequence):
        protein = ProteinAnalysis(sequence)
        return protein.instability_index()

    def calculate_gravy(self, sequence):
        protein = ProteinAnalysis(sequence)
        return protein.gravy()

    def calculate_secondary_structure_fraction(self, sequence):
        protein = ProteinAnalysis(sequence)
        helix, turn, sheet = protein.secondary_structure_fraction()
        return helix, turn, sheet


    def calculate_properties(self, sequence, steric_parameter=True, polarizability=True,
                            volume=True, hydrophobicity=True, isoelectric_point=True,
                            helix_prob=True, sheet_prob=True):
        StericParameters = {
            'A': 1.28, 'G': 0.0, 'V': 3.67, 'L': 2.59, 'I': 4.19, 'F': 2.94, 'Y': 2.94,
            'W': 3.21, 'T': 3.03, 'S': 1.31, 'R': 2.34, 'K': 1.89, 'H': 2.99, 'D': 1.6,
            'E': 1.56, 'N': 1.6, 'Q': 1.56, 'M': 2.35, 'P': 2.67, 'C': 1.77
        }
        Polarizability = {
            'A': 0.05, 'G': 0.0, 'V': 0.14, 'L': 0.19, 'I': 0.19, 'F': 0.29, 'Y': 0.3,
            'W': 0.41, 'T': 0.11, 'S': 0.06, 'R': 0.29, 'K': 0.22, 'H': 0.23, 'D': 0.11,
            'E': 0.15, 'N': 0.13, 'Q': 0.18, 'M': 0.22, 'P': 0.0, 'C': 0.13
        }
        Volume = {
            'A': 1.0, 'G': 0.0, 'V': 3.0, 'L': 4.0, 'I': 4.0, 'F': 5.89, 'Y': 6.47,
            'W': 8.08, 'T': 2.6, 'S': 1.6, 'R': 6.13, 'K': 4.77, 'H': 4.66, 'D': 2.78,
            'E': 3.78, 'N': 2.95, 'Q': 3.95, 'M': 4.43, 'P': 2.72, 'C': 2.43
        }
        Hydrophobicity = {
            'A': 0.31, 'G': 0.0, 'V': 1.22, 'L': 1.7, 'I': 1.8, 'F': 1.79, 'Y': 0.96,
            'W': 2.25, 'T': 0.26, 'S': -0.04, 'R': -1.01, 'K': -0.99, 'H': 0.13, 'D': -0.77,
            'E': -0.64, 'N': -0.6, 'Q': -0.22, 'M': 1.23, 'P': 0.72, 'C': 1.54
        }
        Isoelectric_point = {
            'A': 6.11, 'G': 6.07, 'V': 6.02, 'L': 6.04, 'I': 6.04, 'F': 5.67, 'Y': 5.66,
            'W': 5.94, 'T': 5.6, 'S': 5.7, 'R': 10.74, 'K': 9.99, 'H': 7.69, 'D': 2.95,
            'E': 3.09, 'N': 6.52, 'Q': 5.65, 'M': 5.71, 'P': 6.8, 'C': 6.35
        }
        Helix_prob = {
            'A': 0.42, 'G': 0.13, 'V': 0.27, 'L': 0.39, 'I': 0.3, 'F': 0.3, 'Y': 0.25,
            'W': 0.32, 'T': 0.21, 'S': 0.2, 'R': 0.36, 'K': 0.32, 'H': 0.27, 'D': 0.25,
            'E': 0.42, 'N': 0.21, 'Q': 0.36, 'M': 0.38, 'P': 0.13, 'C': 0.17
        }
        Sheet_prob = {
            'A': 0.23, 'G': 0.15, 'V': 0.49, 'L': 0.31, 'I': 0.45, 'F': 0.38, 'Y': 0.41,
            'W': 0.42, 'T': 0.36, 'S': 0.28, 'R': 0.25, 'K': 0.27, 'H': 0.3, 'D': 0.2,
            'E': 0.21, 'N': 0.22, 'Q': 0.25, 'M': 0.32, 'P': 0.34, 'C': 0.41
        }

        VS_steric = VS_polar = VS_vol = VS_hydro = VS_iso = VS_helix = VS_sheet = 0

        for aa in sequence:
            if steric_parameter:
                VS_steric += StericParameters.get(aa, 2.4)
            if polarizability:
                VS_polar += Polarizability.get(aa, 2.4)
            if volume:
                VS_vol += Volume.get(aa, 2.4)
            if hydrophobicity:
                VS_hydro += Hydrophobicity.get(aa, 2.4)
            if isoelectric_point:
                VS_iso += Isoelectric_point.get(aa, 2.4)
            if helix_prob:
                VS_helix += Helix_prob.get(aa, 2.4)
            if sheet_prob:
                VS_sheet += Sheet_prob.get(aa, 2.4)

        properties = {}
        if steric_parameter:
            properties['steric_parameter'] = VS_steric / len(sequence)
        if polarizability:
            properties['polarizability'] = VS_polar / len(sequence)
        if volume:
            properties['volume'] = VS_vol / len(sequence)
        if hydrophobicity:
            properties['hydrophobicity'] = VS_hydro / len(sequence)
        if isoelectric_point:
            properties['isoelectric_point'] = VS_iso / len(sequence)
        if helix_prob:
            properties['helix_prob'] = VS_helix / len(sequence)
        if sheet_prob:
            properties['sheet_prob'] = VS_sheet / len(sequence)

        return properties
