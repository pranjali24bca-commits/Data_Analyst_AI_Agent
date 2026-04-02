import os
import pandas as pd

# 🔹 Folder containing CSV files
folder_path = "data"

# 🔹 Output file
output_file = "schema_report_whole.txt"

with open(output_file, "w", encoding="utf-8") as f:

    # Loop through all CSV files
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            print(f"\nProcessing: {file_name}")
            f.write(f"\n{'='*60}\n")
            f.write(f"FILE: {file_name}\n")
            f.write(f"{'='*60}\n")

            try:
                df = pd.read_csv(file_path)

                # 🔹 Schema
                f.write("\n--- SCHEMA ---\n")
                print("\n--- SCHEMA ---")

                for col in df.columns:
                    dtype = df[col].dtype
                    line = f"{col} : {dtype}"
                    print(line)
                    f.write(line + "\n")

                # 🔹 Distinct values
                f.write("\n--- DISTINCT VALUES ---\n")
                print("\n--- DISTINCT VALUES ---")

                for col in df.columns:
                    f.write(f"\nColumn: {col}\n")
                    print(f"\nColumn: {col}")

                    unique_vals = df[col].dropna().unique()

                    # Limit to avoid huge output
                    max_values = 100
                    if len(unique_vals) > max_values:
                        unique_vals = unique_vals[:max_values]
                    else:
                        max_values = len(unique_vals)

                    for val in unique_vals:
                        print(f"  {val}")
                        f.write(f"  {val}\n")

                    if len(df[col].dropna().unique()) > max_values:
                        f.write("  ... (truncated)\n")
                        print("  ... (truncated)")

            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                print(error_msg)
                f.write(error_msg + "\n")

print(f"\n✅ Done! Output saved to: {output_file}")