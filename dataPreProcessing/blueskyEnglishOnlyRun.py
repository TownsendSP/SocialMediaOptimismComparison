import pandas as pd

def filter_english_rows(input_file, output_file):
    """
    Filters a CSV file to keep only rows where the language is English.

    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to save the filtered CSV file

    Returns:
    pandas.DataFrame: Filtered DataFrame with only English language rows
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Filter rows where 'lang' column is 'en'
        df_english = df[df['lang'] == 'en']

        # Save the filtered DataFrame to a new CSV file
        df_english.to_csv(output_file, index=False)

        # Print some information about the filtering
        print(f"Original dataset size: {len(df)} rows")
        print(f"English language dataset size: {len(df_english)} rows")
        print(f"Filtered dataset saved to: {output_file}")

        return df_english

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None
    except KeyError:
        print("Error: 'lang' column not found in the CSV file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    input_csv = "your_input_file.csv"
    output_csv = "english_only_data.csv"

    # Call the function to filter and save
    filtered_df = filter_english_rows(input_csv, output_csv)