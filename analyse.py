import pandas as pd

dirty_test_df = pd.read_csv("/home/preethi/projects/pandas-ai-integration/data/Airbnb/missing_values/dirty_test.csv")
print("Dirty test len: ", len(dirty_test_df))

delete_test_df = pd.read_csv("/home/preethi/projects/pandas-ai-integration/data/Airbnb/missing_values/dirty_train.csv")
print("Dirty train len: ", len(delete_test_df))

cleaned_df = pd.read_csv("/home/preethi/projects/pandas-ai-integration/cleaned_df.csv")
print("Cleaned df: ", len(cleaned_df))
# Get the number of columns
