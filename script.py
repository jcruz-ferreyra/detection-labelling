import re
from pathlib import Path


def convert_date_format(directory):
    directory = Path(directory)
    # Updated pattern to capture everything before the 8-digit date
    pattern = re.compile(r"^(.+?)(\d{8})(.*)$")

    for file in directory.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                prefix = match.group(1)  # Everything before the date
                date = match.group(2)  # The 8-digit date (YYYYMMDD)
                suffix = match.group(3)  # Everything after the date

                new_date = date[2:]  # Convert YYYYMMDD to YYMMDD
                new_name = f"{prefix}{new_date}{suffix}"
                new_path = file.with_name(new_name)

                print(f"Renaming: {file.name} -> {new_name}")
                file.rename(new_path)


if __name__ == "__main__":
    convert_date_format(
        "D:/Documentos/Data/projects/cyclist_census/data/detection/raw/cctv_frames/images"
    )
