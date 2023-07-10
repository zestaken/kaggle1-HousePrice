import pandas as pd

data = pd.read_csv("./submission.csv")
data = data['Sold Price']
print(data)
ids = range(47439, len(data)+1+47438) 
print(ids)
submission = pd.DataFrame()
submission['Id']  = ids
submission['Sold Price'] = data
submission.to_csv("submission1.csv", index=None)

