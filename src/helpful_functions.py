'''
Module: helpful_functions.py
Description: This module contains few copiloted functions for algorythm-intensive yet typical tasks.
'''

import pandas as pd

def check_if_all_numbers_present_for_column(df: pd.DataFrame, 
                                            column_name: str, 
                                            print_flag: bool = False, 
                                            report_ok: bool = False) -> bool:
    """
    Check if all numbers from min value to max value are present in the DataFrame in the specified column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    column_name (str): The name of the column to check for missing numbers.
    
    Returns:
    bool: True if all numbers are present, False otherwise.
    Optionally prints the missing numbers if print_flag is set to True.
    """
    max_value = df[column_name].max()
    min_value = df[column_name].min()
    
    if (min_value >= 0) and (max_value > 0):
        max_value = int(round(max_value))
        min_value = int(round(min_value))
    else:
        print('FYI: function "check_if_all_numbers_present_for_column" reports:',
              'column was empty or had not valid values:')
        print(list(df[column_name].head(30)))
        return False
    
    expected_numbers = set(range(min_value, max_value + 1))
    actual_numbers = set(df[column_name].unique())
    
    missing_numbers = expected_numbers - actual_numbers
    if missing_numbers:
        if print_flag:
            if len(missing_numbers) <= 30:
                print(f"Missing numbers in column '{column_name}': {missing_numbers}")
            else:
                print(f"Too many missing numbers in column '{column_name}': {len(missing_numbers)} totally")
        return False
    else:
        if print_flag and report_ok:
            print(f"All numbers from {min_value} to {max_value} are present in column '{column_name}'.")
        return True