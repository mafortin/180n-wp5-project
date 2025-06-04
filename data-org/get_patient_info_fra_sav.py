import pyreadstat
import argparse 


def read_sav(path2sav_file):

    df, meta = pyreadstat.read_sav(path2sav_file)

    return df, meta



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to read a .sav file and extract patient information.")
    parser.add_argument("-i", "--input_sav", required=True, help="Path to the SAV file.")
    parser.add_argument("-c", "--columns", nargs="+", type=str, required=False, help="Exact column names you want to extract/fetch from the dataframe. Can be >1, use space between names if so.")
    parser.add_argument("--read_only", action="store_true", help="Read only the SAV file without any processing.")

    args = parser.parse_args()

    if args.read_only:
        df, meta = read_sav(args.input_sav)
        print("---------------------------------------------------------")
        print("Dataframe extracted from the SAV file (first 10x10): ")
        print(df.iloc[:10, :10])
        print("---------------------------------------------------------")
        print("Relevant meta information extracted from the SAV file: ")
        print("---------------------------------------------------------")
        print("Number of columns: ", meta.number_columns)
        print("---------------------------------------------------------")
        print("Number of rows: ", meta.number_rows)
        print("---------------------------------------------------------")
        print("Patient IDs :", df.Patient_ID.unique())
        print("---------------------------------------------------------")
        print("Total number of patients: ", len(df.Patient_ID.unique()))
        print("---------------------------------------------------------")

    else:
        df, meta = read_sav(args.input_sav)
        print("---------------------------------------------------------")
        print("Dataframe extracted from the SAV file (first 10x10): ")
        print(df.iloc[:10, :10])
        print("---------------------------------------------------------")
        
        if args.columns:

            print("Relevant columns to be extracted from the SAV file: ")
            print(args.columns)
            print("---------------------------------------------------------")

            try:
                valid_columns = [col for col in args.columns if col in df.columns] # Step to ensure only the existing columns are extracted.
                if len(valid_columns) != len(args.columns):
                    print("Warning: Some of the specified columns do not exist in the dataframe.")
                    print("These columns will be excluded from the extraction: ")
                    print(valid_columns)
                else:

                    new_df = df[valid_columns]
                    # Drop rows where the relevant column (defined by the user) is NaN
                    new_df_filtered = new_df.dropna(subset=[col for col in valid_columns if col not in ['Patient_ID', 'Type']])

                print("New dataframe with selected non-empty columns: ")
                print(new_df_filtered) #new_df.head(10)
                print("---------------------------------------------------------")
            except KeyError as e:
                print(f"Error: One or more columns specified do not exist in the dataframe. {e}")

