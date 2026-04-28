import pandas as pd
from datetime import datetime, timezone
import json

with open('all.json', 'r') as file:

    blob = json.load(file)

prices = blob["history"]

for p in prices:
    print(f"{p["t"]} - {datetime.fromtimestamp(p["p"], tz=timezone.utc)}")

