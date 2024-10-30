import time
import sys
import re

class COLOR:
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    RESET = "\u001b[0m"

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