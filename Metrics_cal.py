import pandas as pd
import numpy as np

def get_NRA(df):
    df['NRA'] = 0

    # Handle case with llm/single annotator
    if 'annotator' not in df.columns:
        # Treat single annotator case same as multiple annotators
        for idx, row in df.iterrows():
            # Get rows up to and including current idx
            prev_rows = df[df.index <= idx]
            witness_count = len(prev_rows[prev_rows['outcome_value'] == 'Witness'])
            questioner_count = len(prev_rows[prev_rows['outcome_value'] == 'Questioner']) 
            # Avoid division by zero by checking length
            total_len = len(prev_rows)  # Include current row in total
            # Use loc to properly assign value to dataframe
            witness_score = witness_count 
            questioner_score = questioner_count 
            df.loc[idx, 'NRA'] = (witness_score - questioner_score) / total_len
    else:
        # Original logic for multiple annotators
        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator']==annotator]
            for idx, row in annotator_data.iterrows():
                prev_rows = annotator_data[annotator_data.index <= idx]
                witness_count = len(prev_rows[prev_rows['outcome_value'] == 'Witness'])
                questioner_count = len(prev_rows[prev_rows['outcome_value'] == 'Questioner'])
                total_len = len(prev_rows)
                witness_score = witness_count 
                questioner_score = questioner_count 
                df.loc[idx, 'NRA'] = (witness_score - questioner_score) / total_len

def get_NRBaT(df):
    # Initialize BaT and PaT columns
    df['bat'] = 0
    df['pat'] = 0

    # Handle case with no annotator column
    if 'annotator' not in df.columns:
        # First calculate BaT and PaT for each turn
        for idx, row in df.iterrows():
            pat = 0
            bat = 0
            # Calculate base commitment value
            if row['Committment_value'] == 2:  # beneficial
                bat += 1
            elif row['Committment_value'] == 3:  # neutral
                bat += 0.5
            elif row['Committment_value'] == 1:  # detrimental
                pat += 1
            elif row['Committment_value'] == 4:  # none
                pat += 0.5
    
            # Add manner and relevance factors
            if row['manner_rate'] >= 3:
                pat += 0.4 * (1 if row['Committment_value'] == 2 else 0.5 if row['Committment_value'] == 3 else 0)
                bat += 0.4 * (1 if row['Committment_value'] == 1 else 0)

            if row['relevance_rate'] >= 3:
                pat += 0.4 * (1 if row['Committment_value'] == 2 else 0.5 if row['Committment_value'] == 3 else 0)
                bat += 0.4 * (1 if row['Committment_value'] == 1 else 0)

            # Add quality/honesty factor
            if row['quality_rate'] >= 3:
                pat += 0.2 * (1 if row['Committment_value'] == 2 else 0.5 if row['Committment_value'] == 3 else 0)
            
            df.loc[idx, 'bat'] = bat
            df.loc[idx, 'pat'] = pat

        # deal with consistency:
        for idx, row in df.iterrows():
            if row['consistency_value'] == 1:
                prev_rows = df[df.index <= idx]  # Get all previous rows
                prev_bat_sum = prev_rows['bat'].sum()  # Sum up all previous bat values
                df.loc[idx, 'pat'] = df['pat'][idx] + 0.2 * prev_bat_sum

        # Calculate z-scores and normalized ratios
        cum_bats = []
        cum_pats = []
        
        for idx, row in df.iterrows():
            prev_rows = df.loc[:idx]  # cumulative rows up to this one
            bat_sum = prev_rows['bat'].sum()
            pat_sum = prev_rows['pat'].sum()
            cum_bats.append(bat_sum)
            cum_pats.append(pat_sum)
            df.loc[idx, 'bat_cumsum'] = bat_sum
            df.loc[idx, 'pat_cumsum'] = pat_sum

        # Z-score normalization of cumulative sums
        bat_z = (np.array(cum_bats) - np.mean(cum_bats)) / np.std(cum_bats, ddof=0)
        pat_z = (np.array(cum_pats) - np.mean(cum_pats)) / np.std(cum_pats, ddof=0)

        # Store z-scores
        df['Z_BaT'] = bat_z
        df['Z_PaT'] = pat_z

        # Compute Z-normalized NRBaT
        for idx in df.index:
            numerator = df.loc[idx, 'Z_BaT'] - df.loc[idx, 'Z_PaT']
            denominator = df.loc[idx, 'Z_BaT'] + df.loc[idx, 'Z_PaT']
            if denominator != 0:
                df.loc[idx, 'ZNRBaT'] = numerator / denominator
                df.loc[idx, 'net_ZNRBaT'] = numerator
            else:
                df.loc[idx, 'ZNRBaT'] = 0
                df.loc[idx, 'net_ZNRBaT'] = 0

    else:
        # Original logic for multiple annotators
        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator']==annotator]
            for idx, row in annotator_data.iterrows():
                pat = 0
                bat = 0
                # Calculate base commitment value
                if row['Committment_value'] == 2:  # beneficial
                    bat += 1
                elif row['Committment_value'] == 3:  # neutral
                    bat += 0.5
                elif row['Committment_value'] == 1:  # detrimental
                    pat += 1
                elif row['Committment_value'] == 4:  # none
                    pat += 0.5
        
                # Add manner and relevance factors
                if row['manner_rate'] >= 3:
                    pat += 0.4 * (1 if row['Committment_value'] == 2 else 0.5 if row['Committment_value'] == 3 else 0)
                    bat += 0.4 * (1 if row['Committment_value'] == 1 else 0) # compensation does not happen for none commitment

                if row['relevance_rate'] >= 3:
                    pat += 0.4 * (1 if row['Committment_value'] == 2 else 0.5 if row['Committment_value'] == 3 else 0)
                    bat += 0.4 * (1 if row['Committment_value'] == 1 else 0)

                # Add quality/honesty factor
                if row['quality_rate'] >= 3:
                    pat += 0.2 * (1 if row['Committment_value'] == 2 else 0.5 if row['Committment_value'] == 3 else 0)
                
                    
                df.loc[idx, 'bat'] = bat
                df.loc[idx, 'pat'] = pat
        
        # deal with consistency:
        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator']==annotator]
            for idx, row in annotator_data.iterrows():
                if row['consistency_value'] == 1:
                    prev_rows = annotator_data[annotator_data.index <= idx]  # Get all previous rows
                    prev_bat_sum = prev_rows['bat'].sum()  # Sum up all previous bat values
                    # pat += 0.5 * prev_bat_sum
                    df.loc[idx, 'pat'] = df['pat'][idx] + 0.2 * prev_bat_sum

        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator'] == annotator].copy()
            cum_bats = []
            cum_pats = []
            
            for idx, row in annotator_data.iterrows():
                prev_rows = annotator_data.loc[:idx]  # cumulative rows up to this one
                bat_sum = prev_rows['bat'].sum()
                pat_sum = prev_rows['pat'].sum()
                cum_bats.append(bat_sum)
                cum_pats.append(pat_sum)
                df.loc[idx, 'bat_cumsum'] = bat_sum
                df.loc[idx, 'pat_cumsum'] = pat_sum

            # Z-score normalization of cumulative sums
            bat_z = (np.array(cum_bats) - np.mean(cum_bats)) / np.std(cum_bats, ddof=0)
            pat_z = (np.array(cum_pats) - np.mean(cum_pats)) / np.std(cum_pats, ddof=0)

            # Store z-scores
            df.loc[annotator_data.index, 'Z_BaT'] = bat_z
            df.loc[annotator_data.index, 'Z_PaT'] = pat_z

            # Compute Z-normalized NRBaT
            for i, idx in enumerate(annotator_data.index):
                numerator = bat_z[i] - pat_z[i]
                denominator = bat_z[i] + pat_z[i]
                if denominator != 0:
                    df.loc[idx, 'ZNRBaT'] = numerator / denominator
                    df.loc[idx, 'net_ZNRBaT'] = numerator
                else:
                    df.loc[idx, 'ZNRBaT'] = 0  # or np.nan if you prefer