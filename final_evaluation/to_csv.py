import argparse
import os


def to_csv(filename):
    with open(filename, "r") as txt_file:
        csv_list = []
        for line in txt_file:
            line = line.strip("\n")
            if "=====" in line:
                if csv_list:
                    _write_csv(filename, ",".join(csv_list))
                    csv_list = []
                csv_list.append(line.strip("="))
            if "number_crashes" in line:
                csv_list.append(line.split()[-1])
            if "time_to_finish" in line:
                csv_list.append(str(round(float(line.split()[-1]), ndigits=2)))
            if "Reason" in line:
                csv_list.append(line.split()[-1])
                csv_list.append("60")


def _write_csv(filename, line):
    with open(filename[:-3] + "csv", "a+") as csv_file:
        csv_file.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bring summary to CSV")
    parser.add_argument(
        "--filename",
        help="Name of the summary .txt file",
        required=True
    )

    args = parser.parse_args()
    to_csv(filename=args.filename)
