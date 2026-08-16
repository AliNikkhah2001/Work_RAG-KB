import pandas as pd
import os

path = r'C:\Users\10225\Downloads\KB\extracted'

# Read a few key Excel files to understand structure
files_to_check = [
    r'توضیحات تشریحی ریزن کدها\Reason codes structure for chatbot.xlsx',
    r'توضیحات تشریحی ریزن کدها\حقوقی\ReasonCode_Company_Chatbot.xlsx',
    r'توضیحات تشریحی ریزن کدها\حقیقی\ReasonCode_IndividualMain-chatbot.xlsx',
    r'توضیحات تشریحی ریزن کدها\چک\Cheque_ReasonCode_for_Chatbot.xlsx',
    r'سوالات پیشنهادی.xlsx',
    r'مقالات کاربردی\محمدی\مفاهیم پایه\AIandCreditScoring.xlsx',
    r'‫سوال و جواب‌های CRM\حقوقی\Company_CRM_Questions.xlsx',
    r'‫سوال و جواب‌های CRM\چک\ChequeQuestions.xlsx',
    r'آیین نامه‌ها\سوالات پیشنهادی.xlsx',
]

for f in files_to_check:
    full_path = os.path.join(path, f)
    if os.path.exists(full_path):
        print(f"\n{'='*80}")
        print(f"FILE: {f}")
        print(f"{'='*80}")
        try:
            df = pd.read_excel(full_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Dtypes:\n{df.dtypes}")
            print(f"First 3 rows:\n{df.head(3).to_string()}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\nNOT FOUND: {f}")