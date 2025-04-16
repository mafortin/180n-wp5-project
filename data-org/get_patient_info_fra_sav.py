import pyreadstat
import argparse 


def read_sav(path2sav_file):

    df, meta = pyreadstat.read_sav(path2sav_file)

    return df, meta



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to read a .sav file and extract patient information.")
    parser.add_argument("-i", "--input_sav", required=True, help="Path to the SAV file.")

    args = parser.parse_args()


    df, meta = read_sav(args.input_sav)
    print("---------------------------------------------------------")
    print("Dataframe extracted from the SAV file (first 10x10): ")
    print(df.iloc[:10, :10])
    print("---------------------------------------------------------")
    print("Relevant meta information extracted from the SAV file: ")
    print("---------------------------------------------------------")
    #print("Column names: ", meta.column_names)
    print("Number of columns: ", meta.number_columns)
    print("---------------------------------------------------------")
    print("Number of rows: ", meta.number_rows)
    print("---------------------------------------------------------")
    print("Patient IDs :", df.Patient_ID.unique())
    print("---------------------------------------------------------")
    print("Total number of patients: ", len(df.Patient_ID.unique()))
    print("---------------------------------------------------------")

