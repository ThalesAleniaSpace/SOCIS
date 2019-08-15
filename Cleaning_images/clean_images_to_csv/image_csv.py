

import pandas as pd
df = pd.read_csv("values.csv")

col = '_file_'


for index, row in df.iterrows():
    rep = str("" + str(row["dir"]) + "_"+str(row["id"])+".BMP")
    print(rep)
    row[col]  = rep
    df.at[index,col] = rep

df.to_csv('image_clean.csv', index=False, sep=',')