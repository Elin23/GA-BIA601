import pandas as pd
def read_dataset(file,target):
    data=pd.read_csv(file)
    
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    x=data.drop(columns=[target]).values
    y=data[target].values
    return x,y