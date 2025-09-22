
import pandas as pd
import re
import numpy as np
def clean_and_convert_to_array(data):

    cleaned_data = data.replace("[", "").replace("]", "").replace("\n", " ")
    
    try:
        array = np.array([float(x) for x in cleaned_data.split()])
        return array
    except ValueError as e:
        return f"transform error: {e}"
def data_processing(data):

    data.iloc[:, 2:4] = data.iloc[:, 2:4].applymap(clean_and_convert_to_array)
    return data 




