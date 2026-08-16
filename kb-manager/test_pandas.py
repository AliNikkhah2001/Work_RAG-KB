import pandas as pd

df = pd.read_excel(r"C:\Users\10225\Downloads\KB\extracted\اشکالات سامانه اعتباریتو\EtebaritoProblems.xlsx")
with open("pandas_test.txt", "w", encoding="utf-8") as f:
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(f"Shape: {df.shape}\n")
    for i, row in df.head(2).iterrows():
        f.write(f"---ROW {i}---\n")
        for col in df.columns:
            val = row[col]
            if pd.notna(val):
                f.write(f"  {col}: {repr(str(val))[:200]}\n")