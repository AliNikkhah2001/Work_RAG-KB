import pandas as pd

# Try different engines
engines = ['openpyxl', 'xlrd', 'odf', 'pyxlsb']

for engine in engines:
    try:
        df = pd.read_excel(r"C:\Users\10225\Downloads\KB\extracted\اشکالات سامانه اعتباریتو\EtebaritoProblems.xlsx", engine=engine)
        with open(f"pandas_{engine}.txt", "w", encoding="utf-8") as f:
            f.write(f"Engine: {engine}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            f.write(f"Shape: {df.shape}\n")
            for i, row in df.head(2).iterrows():
                f.write(f"---ROW {i}---\n")
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        f.write(f"  {col}: {repr(str(val))[:200]}\n")
        print(f"Engine {engine}: SUCCESS")
    except Exception as e:
        with open(f"pandas_{engine}.txt", "w", encoding="utf-8") as f:
            f.write(f"Engine: {engine}\n")
            f.write(f"ERROR: {e}\n")
        print(f"Engine {engine}: FAILED - {e}")

print("Done")