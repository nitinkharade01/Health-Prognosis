import csv
import random

header = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","target"
]

def random_row():
    return [
        random.randint(29, 77),            # age
        random.randint(0, 1),              # sex
        random.randint(0, 3),              # cp
        random.randint(94, 200),           # trestbps
        random.randint(149, 564),          # chol
        random.randint(0, 1),              # fbs
        random.randint(0, 2),              # restecg
        random.randint(88, 202),           # thalach
        random.randint(0, 1),              # exang
        round(random.uniform(0.0, 6.2), 1),# oldpeak
        random.randint(0, 2),              # slope
        random.randint(0, 4),              # ca
        random.randint(1, 3),              # thal
        random.randint(0, 1)               # target
    ]

# Read existing data
with open('datasets/heart.csv', 'r', newline='') as f:
    reader = list(csv.reader(f))
    existing = reader[1:]  # skip header

# Generate 500 new rows
rows = existing + [random_row() for _ in range(500)]

# Write all rows back (with header)
with open('datasets/heart.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print(f"Generated {len(rows)} rows in datasets/heart.csv.") 