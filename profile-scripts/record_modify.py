import csv
import argparse

parser = argparse.ArgumentParser(description="A simple example program.")
parser.add_argument("--file", type=str, required=False, default="./profile-output/ggml_profile_node_records.csv")
args = parser.parse_args()

profile_record_path = args.file
output_base_name = profile_record_path.split(".csv")[0]

def modify_csv_column(input_file, output_file, modifier):
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            node_compute_time_ns = float(row[4])
            row[4] = str(modifier * node_compute_time_ns)

            writer.writerow(row)

modifiers = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

for mod in modifiers:
    output_name = output_base_name + f"-{str(mod)}.csv"
    modify_csv_column(profile_record_path, output_name, mod)
