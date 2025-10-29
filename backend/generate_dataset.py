import numpy as np
import pandas as pd

num_rows = 200
num_cols = 40

data = np.random.randint(0, 101, size=(num_rows, num_cols))

columns = [f"feature_{i+1}" for i in range(num_cols)]

df = pd.DataFrame(data, columns=columns)

output_file = "big_dataset.csv"
df.to_csv(output_file, index=False)

print(f" Dataset CSV : {output_file}")
