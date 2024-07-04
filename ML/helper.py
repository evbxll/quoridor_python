import time
import sys
import re

class COLOR:
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    RESET = "\u001b[0m"

def progress_bar(current, total):
    # Clear previous progress bar (move cursor up one line and clear it)
    if current > 1:
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.write("\033[K")  # Clear current line

    # Calculate percentage progress
    progress_percent = round(100 * current / total)

    # Print the progress bar with current and total count
    sys.stdout.write(f"Progress: [{'#' * progress_percent}{' ' * (100 - progress_percent)}] ")
    sys.stdout.write(f"{current}/{total} ({progress_percent}%)")
    sys.stdout.flush()
    sys.stdout.write("\n")  # Move to the next line after completing the progress update


def getNumRounds(file_path):
    pattern = r"rounds_(\d+)"
    # Use re.search to find the match
    match = re.search(pattern, file_path)

    if match:
        # Extract the number after "rounds_"
        rounds_number = int(match.group(1))
        print("Number of rounds:", rounds_number)
    else:
        print("No match found for rounds number.")

    return rounds_number