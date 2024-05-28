import pandas as pd

csv_files_path = "csv_files/"
raw_file_name = "diabetes.csv"
output_file_name = "cleaned_diabetes.csv"

df = pd.read_csv(r"D:\MiningFinal\project\csv_files\diabetes.csv")

# drop rows with missing values or 0 values in all columns except Outcome column
df = df[
    (df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] != 0).all(
        axis=1
    )
]

# drop rows with missing values in Outcome column
df = df.dropna(subset=["Outcome"])

# drop dubplicates rows
df = df.drop_duplicates()

# Interquartile range method to remove outliers in all columns except Outcome column
Q1 = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].quantile(0.25)
Q3 = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].quantile(0.75)
IQR = Q3 - Q1
df = df[
    ~(
        (
            df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]]
            < (Q1 - 1.5 * IQR)
        )
        | (
            df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]]
            > (Q3 + 1.5 * IQR)
        )
    ).any(axis=1)
]

# Encoding Categorical Data
df["Outcome"] = df["Outcome"].map({1: "Diabetic", 0: "Not Diabetic"})

# !Quality Assurance Checks
# check for missing values
print(df.isnull().sum())

# check for 0 values
print((df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] == 0).sum())

# check for duplicates
print(df.duplicated().sum())

# check for outliers
print(df.describe())

# check the number of rows and columns
print(df.shape)

# check the first 5 rows of the data
print(df.head())

# save the cleaned data to the file
df.to_csv(r"D:\MiningFinal\project\csv_files\cleaned_diabetes.csv", index=False)
