import os
from typing import List
NAMES_PATH = os.path.join('.', 'names.txt')

def read_names() -> List[str]:
    with open(NAMES_PATH, 'r') as f:
        names = f.read().splitlines()
    return names
