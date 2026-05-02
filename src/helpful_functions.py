'''
Module: helpful_functions.py
Description: This module contains few copiloted functions for algorythm-intensive yet typical tasks.
'''

import pandas as pd

def check_if_all_numbers_present_for_column(df: pd.DataFrame, column_name: str, print_flag: bool = False) -> bool:
    """
    Check if all numbers from 0 to (len(df)-1) value are present in the DataFrame in the specified column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    column_name (str): The name of the column to check for missing numbers.
    
    Returns:
    bool: True if all numbers are present, False otherwise.
    Optionally prints the missing numbers if print_flag is set to True.
    """
    max_value = df[column_name].max()
    expected_numbers = set(range(max_value + 1))
    actual_numbers = set(df[column_name].unique())
    
    missing_numbers = expected_numbers - actual_numbers
    if missing_numbers:
        if print_flag:
            print(f"Missing numbers in column '{column_name}': {missing_numbers}")
        return False
    else:
        if print_flag:
            print(f"All numbers from 0 to {max_value} are present in column '{column_name}'.")
        return True