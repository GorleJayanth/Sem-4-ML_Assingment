import pandas as pd

def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# Example Usage
file_path = "Lab Session Data.xlsx"
sheet_name = "IRCTC Stock Price"
df = load_data(file_path, sheet_name)
print(df.head())  # Display first few rows
