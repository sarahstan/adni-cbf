import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_all_csvs(data_folder='data'):
    """
    Load all CSV files from the specified folder into a dictionary of DataFrames
    Returns: Dictionary with filenames as keys and DataFrames as values
    """
    dataframes = {}
    
    # List all CSV files in the data folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # Load each CSV file
    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            dataframes[file] = df
            print(f"Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    return dataframes

def merge_simple_data(master_df, new_df, columns_to_add, prefix=None):
    """
    Merge data that only needs RID matching
    
    Parameters:
    - master_df: Master dataframe
    - new_df: Dataframe to merge in
    - columns_to_add: List of column names to add from new_df
    - prefix: Optional prefix for new column names
    """
    cols_to_merge = ['RID'] + columns_to_add
    df_to_merge = new_df[cols_to_merge]
    
    if prefix:
        df_to_merge = df_to_merge.rename(
            columns={col: f"{prefix}_{col}" for col in columns_to_add}
        )
    
    merged_df = master_df.merge(
        df_to_merge,
        on='RID',
        how='left'
    )
    
    return merged_df

def merge_temporal_data(master_df, new_df, columns_to_add, prefix=None):
    """
    Merge data that needs both RID and VISCODE matching
    """
    cols_to_merge = ['RID', 'VISCODE'] + columns_to_add
    df_to_merge = new_df[cols_to_merge]
    
    if prefix:
        df_to_merge = df_to_merge.rename(
            columns={col: f"{prefix}_{col}" for col in columns_to_add}
        )
    
    merged_df = master_df.merge(
        df_to_merge,
        on=['RID', 'VISCODE'],
        how='left'
    )
    
    return merged_df

def add_progression_column(df):
    """
    Adds a 'Progression' column to the dataframe based on how diagnosis changes over time for each subject.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing columns 'RID', 'VISCODE', and 'DX'
    
    Returns:
    --------
    pandas.DataFrame
        The original dataframe with a new 'Progression' column added
    """
    # Make a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Create a new column for Progression with default value
    df_result['Progression'] = None
    
    # Get unique subjects
    unique_subjects = df_result['RID'].unique()
    
    for subject in unique_subjects:
        # Get data for this subject, sorted by VISCODE
        subject_data = df_result[df_result['RID'] == subject].sort_values(by='VISCODE')
        
        # Extract diagnoses, removing any missing values (NaN)
        diagnoses = subject_data['DX'].dropna().tolist()
        
        # Skip if no diagnoses or just one diagnosis timepoint
        if len(diagnoses) <= 1:
            continue
            
        # Check if all diagnoses are the same
        if all(dx == diagnoses[0] for dx in diagnoses):
            if len(diagnoses) >= 3:  # Stable for at least 3 timepoints
                if diagnoses[0] == 'CN':
                    progression = 'StableCN'
                elif diagnoses[0] == 'MCI':
                    progression = 'StableMCI'
                elif diagnoses[0] == 'Dementia':
                    progression = 'StableDementia'
                else:
                    progression = 'Unknown'  # For any other diagnosis value
            else:
                progression = 'Insufficient'  # Not enough timepoints
        else:
            # Check for various progression patterns
            # Convert diagnoses to a string to easily check patterns
            dx_str = ''.join(['C' if dx == 'CN' else 'M' if dx == 'MCI' else 'D' for dx in diagnoses])
            
            # Look for specific patterns
            if 'CM' in dx_str and 'MC' not in dx_str and 'D' not in dx_str:
                progression = 'CNtoMCI'
            elif 'CMD' in dx_str and 'DC' not in dx_str and 'DM' not in dx_str:
                progression = 'CNtoMCItoDementia'
            elif 'MD' in dx_str and 'CM' not in dx_str and 'DM' not in dx_str and 'DC' not in dx_str:
                progression = 'MCItoDementia'
            elif 'MC' in dx_str and 'CM' not in dx_str and 'D' not in dx_str:
                progression = 'MCIrecovery'
            else:
                progression = 'Chaotic'
                
        # Assign progression value to all rows for this subject
        df_result.loc[df_result['RID'] == subject, 'Progression'] = progression
        
    return df_result

# Example usage:
# df_filtered = add_progression_column(df_filtered)

def add_diagnosis_onset_columns(df):
    """
    Adds columns containing the month number for the first occurrence of each diagnosis type
    for each subject. Creates a 'Months' column by converting VISCODE to numerical months.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing columns 'RID', 'VISCODE', and 'DX'
    
    Returns:
    --------
    pandas.DataFrame
        The original dataframe with new 'Months' and onset columns
    """
    # Make a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Create Months column by converting VISCODE to numerical months
    def viscode_to_months(viscode):
        if pd.isna(viscode):
            return np.nan
        
        viscode = str(viscode).lower()
        if viscode == 'bl':
            return 0
        elif viscode.startswith('m'):
            # Extract the number after 'm'
            try:
                months = int(viscode[1:])
                return months
            except ValueError:
                return np.nan
        else:
            return np.nan
    
    # Apply the conversion to create the Months column
    df_result['Months'] = df_result['VISCODE'].apply(viscode_to_months)
    
    # Create new onset columns with NaN values as default
    df_result['CNonset'] = np.nan
    df_result['MCIonset'] = np.nan
    df_result['Dementiaonset'] = np.nan
    
    # Get unique subjects
    unique_subjects = df_result['RID'].unique()
    
    for subject in unique_subjects:
        # Get data for this subject, sorted by Months
        subject_data = df_result[df_result['RID'] == subject].sort_values(by='Months')
        
        # Find first occurrence of each diagnosis type
        for diagnosis in ['CN', 'MCI', 'Dementia']:
            # Check if this diagnosis exists for this subject
            diagnosis_rows = subject_data[subject_data['DX'] == diagnosis]
            if not diagnosis_rows.empty:
                # Get the Months value of the first occurrence
                first_months = diagnosis_rows.iloc[0]['Months']
                
                # Set the corresponding onset column for ALL rows of this subject
                column_name = f"{diagnosis}onset"
                df_result.loc[df_result['RID'] == subject, column_name] = first_months
    
    return df_result
# Example usage:
# df_filtered = add_diagnosis_onset_columns(df_filtered)

def convert_to_numeric(df, column_list):
    """
    Convert specified columns from string to numeric (float64), handling '<' and '>' characters.
        For ABETA (>1700) becomes 1800. 
        TAU (<80) becomes 70. 
        PTAU (<8) becomes 3. 
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the columns to convert
    column_list (list): List of column names to convert to numeric
    
    Returns:
    pandas.DataFrame: DataFrame with converted columns
    """
    import pandas as pd
    import numpy as np
    import re
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    for column in column_list:
        if column in result_df.columns:
            # Function to clean strings and convert to numeric
            def clean_and_convert(value):
                if pd.isna(value):
                    return np.nan
                
                # Convert to string if not already
                if not isinstance(value, str):
                    return float(value)
                
                # Remove '<', '>', and any other non-numeric characters except decimal point
                # Keep the numeric part
                numeric_part = re.sub(r'[^0-9.-]', '', value)
                # Convert to float first for comparison
                try:
                    numeric_float = float(numeric_part) if numeric_part else np.nan
                    
                    # Adjust the numeric part according to rules
                    if numeric_float == 1700:
                        numeric_float = 1800
                    elif numeric_float == 80:
                        numeric_float = 70
                    elif numeric_float == 8:
                        numeric_float = 5
                    
                    return numeric_float
                except ValueError:
                    return np.nan
                            
            # Apply the conversion function and convert to float64
            result_df[column] = result_df[column].apply(clean_and_convert).astype('float64')
            print(f"Converted '{column}' to numeric float64")

        else:
            print(f"Warning: Column '{column}' not found in the dataframe")
    
    return result_df

# Example usage:
# columns_to_convert = ['ABETA', 'TAU', 'PTAU']
# df_filtered_logical = convert_to_numeric(df_filtered_logical, columns_to_convert)

def fill_consistent_dx(df, rid_col='RID', dx_col='DX', month_col='Month'):
    """
    Fill missing DX values only when:
    1. Previous and next non-missing DX values are the same
    2. There are at most 2 consecutive missing values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing patient, diagnosis and session information
    rid_col : str, default='RID'
        Column name for patient ID
    dx_col : str, default='DX'
        Column name for diagnosis
    month_col : str, default='Month'
        Column name for session month/time
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with filled DX values according to the criteria
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Group by patient ID and sort by month within each group
    for rid, patient_data in result_df.groupby(rid_col):
        # Sort by month to ensure correct temporal order
        patient_data = patient_data.sort_values(by=month_col)
        
        # Get indices of the patient data in the original dataframe
        indices = patient_data.index
        dx_values = patient_data[dx_col].values
        
        i = 0
        while i < len(dx_values):
            # Skip non-missing values
            if pd.notna(dx_values[i]):
                i += 1
                continue
                
            # Found a missing value, find the previous non-missing value
            prev_idx = i - 1
            while prev_idx >= 0 and pd.isna(dx_values[prev_idx]):
                prev_idx -= 1
                
            # If no previous non-missing value, or too many missing values, skip
            if prev_idx < 0:
                i += 1
                continue
                
            prev_dx = dx_values[prev_idx]
            
            # Find the next non-missing value
            next_idx = i + 1
            while next_idx < len(dx_values) and pd.isna(dx_values[next_idx]):
                next_idx += 1
                
            # If no next non-missing value, or too many missing values, skip
            if next_idx >= len(dx_values) or (next_idx - prev_idx > 3):
                i += 1
                continue
                
            next_dx = dx_values[next_idx]
            
            # If previous and next DX values match, fill the missing values
            if prev_dx == next_dx:
                # Fill only if there are at most 2 missing values
                if next_idx - prev_idx <= 3:  # (prev, [missing1, missing2], next)
                    for j in range(prev_idx + 1, next_idx):
                        result_df.loc[indices[j], dx_col] = prev_dx
            
            i = next_idx
    
    return result_df

def process_all_data():
    """
    Main function to load and process all data
    """
    # Load all CSV files
    all_dataframes = load_all_csvs()
    
    # Start with master dataframe
    master_df = all_dataframes['ADNIMERGE_27Jan2025.csv']
    final_df = master_df.copy()
    
    # Example merge (you can add more as needed)
    file1_df = all_dataframes['DESIKANLAB_27Jan2025.csv']  # Replace with actual filename
    final_df = merge_simple_data(
        final_df,
        file1_df,
        columns_to_add=['PHS', 'CIR'],
        prefix='Gen'
    )
    
    # Example: Merge temporal data
    file2_df = all_dataframes['UCSFSNTVOL_27Jan2025.csv']
    final_df = merge_temporal_data(
        final_df,
        file2_df,
        columns_to_add=['LEFTHIPPO','RIGHTHIPPO'],
        prefix='VOL'
    )
    # Calculate median age, create new column based on median split
    age_median = final_df['AGE'].median()
    final_df['AGE_Med'] = np.where(final_df['AGE'] > age_median, 'Older', 'Younger')
    # Fill missing DX values (if one or two missing, and prior and post match)
    final_df = fill_consistent_dx(final_df)
    # Create filtered dataframe removing rows without a diagnosis
    filtered_df = final_df.dropna(subset=['DX'])   
    # Add progression column to subjects that have a diagnosis
    print("About to call add_progression_column")
    filtered_df = add_progression_column(filtered_df)   
    print("Ran add_progression_column")
    #Update filtered dataframes to exclude chaotic progression
    filtered_df=filtered_df[filtered_df['Progression']!='Chaotic']
    print("Excluded Chaotic progression")
    #Create filtered dataframe removing visits with missing Hippocampus CBF data
    filtered_cbf=filtered_df.dropna(subset=['Hippocampus'])
    #Add diagnosis onset columns to filtered dataframes
    filtered_cbf = add_diagnosis_onset_columns(filtered_cbf)
    filtered_df = add_diagnosis_onset_columns(filtered_df)
    #Convert ABETA, TAU, PTAU to numeric
    columns_to_convert = ['ABETA', 'TAU', 'PTAU']
    filtered_cbf = convert_to_numeric(filtered_cbf, columns_to_convert)
    filtered_df = convert_to_numeric(filtered_df, columns_to_convert)
    #Create filtered_cbf_logical to do survival analyses by excluding MCIrecovery
    filtered_cbf_logical=filtered_cbf[filtered_cbf['Progression']!='MCIrecovery']
    filtered_df_logical = filtered_df[filtered_df['Progression']!='MCIrecovery']
    #Separate filtered_df into datasets for modeling with AI/ML
    # Dataset 1: StableCN and CNtoMCI patients
    filtered_df_forCN = filtered_df[filtered_df['Progression'].isin(['StableCN', 'CNtoMCI'])]
    # Dataset 2: StableMCI and MCItoDementia patients
    filtered_df_forMCI = filtered_df[filtered_df['Progression'].isin(['StableMCI', 'MCItoDementia'])]
    # Dataset 3: MCIRecovery patients
    filtered_df_forRecovery = filtered_df[filtered_df['Progression'] == 'MCIRecovery']

    return (final_df, filtered_df, filtered_cbf, filtered_cbf_logical,filtered_df_logical,\
            filtered_df_forCN,filtered_df_forMCI,filtered_df_forRecovery)

if __name__ == '__main__':
    final_df, filtered_df, filtered_cbf, filtered_cbf_logical,filtered_df_logical,\
        filtered_df_forCN,filtered_df_forMCI,filtered_df_forRecovery = process_all_data()
    print("\nFinal DataFrame shape:", final_df.shape)
    print("\nFinal DataFrame columns:", final_df.columns)
    # Check how many rows remain
    print(f"Shape after filtering: {filtered_df.shape}")
    # Look at distribution of DX values to verify
    print("\nDX value counts:")
    print(filtered_df['DX'].value_counts())

    
