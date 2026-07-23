import csv
import glob

csv_files = glob.glob("./data/catalog/ps1/*.csv")
output_file = "./data/merged_ps1.csv"

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)

    for i, file_name in enumerate(csv_files):
        print(f'Merging {file_name}')
        with open(file_name, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)

            # Extract the header row
            header = next(reader)

            # Write header only for the very first file
            if i == 0:
                writer.writerow(header)

            # Stream the rest of the rows directly to the output file
            for row in reader:
                
                writer.writerow(row)