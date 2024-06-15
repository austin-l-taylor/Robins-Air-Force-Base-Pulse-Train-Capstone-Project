import pandas as pd
import glob


def main():
    del_row = "BUFFER"
    file_list = glob.glob("./PulseTrainData/*.csv")

    for file in file_list:
        df = pd.read_csv(file)  # Corrected: should be pd.read_csv

        # Remove rows where any column contains 'BUFFER'
        df_clean = df[
            ~df.apply(lambda row: row.astype(str).str.contains(del_row).any(), axis=1)
        ]

        # Write the cleaned DataFrame back to the same CSV file
        df_clean.to_csv(file, index=False)


if __name__ == "__main__":
    main()
