import csv
import random
import time

class RandomRowSelector:

    """
    A class to randomly select rows from a CSV file and write them to another CSV file.
    """

    def __init__(self, input_csv, output_csv, num_rows):
        # initialize time to start measuring
        start_time = time.time()

        self.input_csv = input_csv
        self.output_csv = output_csv
        self.num_rows = num_rows

        self.select_random_rows()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for whole process: {elapsed_time} seconds")


    def select_random_rows(self):
        """
        This method randomly selects a specified amount of rows from a csv file
        and writes them to the output CSV file.
        
        :param num_rows: Number of rows to randomly select
        :raises ValueError: If the number of rows requested exceeds available rows
        """
        # Read all rows from the input CSV
        with open(self.input_csv, 'r') as infile:
            reader = list(csv.reader(infile))
            
            # Extract the header and data rows
            # header = reader[0]
            # data_rows = reader[1:]
            # no header:
            data_rows = reader
            
            # Ensure num_rows does not exceed the available data rows
            if self.num_rows > len(data_rows):
                raise ValueError(f"Requested {self.num_rows} rows, but only {len(data_rows)} available.")
            
            # Randomly sample the specified number of rows
            selected_rows = random.sample(data_rows, self.num_rows)
            
            # Write the selected rows to the output CSV, including the header
            with open(self.output_csv, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                # writer.writerow(header)  # Write the header first
                writer.writerows(selected_rows)  # Write the selected rows

if __name__ == "__main__":

    print("Main block in random_row_sampling.py executed!")

    # Example usage:
    input_csv = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels_EF_categories/all_random_val_test/ED_cat_all_frames.csv'  # Path to the all frames input CSV file
    output_csv = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/labels/random_1_percent_again.csv'  # Path to the output CSV file
    num_rows = 13153  # Number of rows to randomly select

    RandomRowSelector(input_csv, output_csv, num_rows)