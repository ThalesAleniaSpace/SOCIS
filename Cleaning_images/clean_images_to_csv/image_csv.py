

import pandas as pd
df = pd.read_csv("values.csv")

col = '_file_'


for index, row in df.iterrows():
    l = str(row["dir"])
    if len(l) == 1 :
        row["dir"]= "00"+l
    if len(l) == 2 :
        row["dir"] = "0"+l
    rep = str("" + str(row["id"]) + "_"+str(row["dir"])+".bmp")
    print(rep)
    row[col]  = rep
    df.at[index,col] = rep

df.to_csv('image_clean.csv', index=False, sep=',')