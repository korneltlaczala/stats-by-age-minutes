import os
import pandas as pd
import openpyxl

DATA_DIR = "data/"
FILES_TO_PROCESS = [os.path.join(DATA_DIR, f"U{age}.xlsx") for age in range(14, 20)]

def process_file(file_path):

    if not os.path.exists(file_path):
        print(f"⚠️  File {file_path} does not exist. Skipping.")
        return None
    print(f"📊 Processing file: {file_path}...")

    try:
        df = pd.read_excel(file_path, header=None)

        required_col_count = 19
        if df.shape[1] < required_col_count:
            for i in range(df.shape[1], required_col_count):
                df[i] = pd.NA

        print(df.head())
        df = df.dropna(subset=[1])
        df_clean = df.iloc[:, [1, 6, 8, 10, 14, 16, 18]].copy()
        df_clean.columns = ['Name', 'DOB', 'Q_birth', 'Minutes', '5m', '30m', 'Beep']
        df_clean["Age_group"] = os.path.basename(file_path).split('.')[0]
        return df_clean

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


all_dataframes = []
for file in FILES_TO_PROCESS:
    data = process_file(file)
    os.makedirs("processed/", exist_ok=True)
    if data is not None:
        all_dataframes.append(data)
        output_file = file.replace("data/", "processed/")
        data.to_excel(output_file, index=False)
        print(f"✅ Processed data saved to {output_file}")

if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_output_file = os.path.join("processed/", "combined_data.xlsx")
    combined_df.to_excel(combined_output_file, index=False)
    combined_df.to_csv(combined_output_file.replace('.xlsx', '.csv'), index=False)
    print(f"✅ Combined data saved to {combined_output_file}")
