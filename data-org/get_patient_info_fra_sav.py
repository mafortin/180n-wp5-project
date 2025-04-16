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
    print("Dataframe extracted from the SAV file: ", df)
    print("---------------------------------------------------------")
    print("Meta information extracted from the SAV file: ", meta)
    print("---------------------------------------------------------")

