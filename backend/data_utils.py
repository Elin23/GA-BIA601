import pandas as pd
def read_dataset(file,target):
    data=pd.read_csv(file)
    x=data.drop(columns=[target]).values
    y=data[target].values
    return x,y