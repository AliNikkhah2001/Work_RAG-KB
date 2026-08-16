import pandas as pd
import os

path = r'C:\Users\10225\Downloads\KB\extracted'
f = r'توضیحات تشریحی ریزن کدها\Reason codes structure for chatbot.xlsx'
full_path = os.path.join(path, f)
df = pd.read_excel(full_path, nrows=5)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print(df.head(5).to_string())