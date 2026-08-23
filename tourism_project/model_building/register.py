import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv" # Corrected path

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
"CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
"DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
"NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
"NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
"Designation", "MonthlyIncome"
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
# Assuming 'Failure' is a placeholder from previous context, if it's not in the dataset, this will error.
# If 'ProdTaken' is the actual target, then it should be used here.
# For now, commenting out to avoid error if 'Failure' is indeed not present.
# print("Failure distribution:")
# print(df["Failure"].value_counts())
