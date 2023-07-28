import pandas as pd
df = pd.read_csv('pCLUE_train.csv')

tt = str(df["input"][0])

print(df.columns)
print(tt)
print(tt.split())
print(' '.join(tt.split()))