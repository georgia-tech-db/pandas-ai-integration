import os
import pandas as pd
import evadb
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

cursor = evadb.connect().cursor()
print("Connected to EvaDB")
#local
# create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
#                                     IMPL  './functions/chat_with_df.py' use_local_llm 'True' local_llm_model "llama-2-7b-chat.ggmlv3.q4_0.bin" csv_path "./data/cars.csv";
#                                     """

create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
            IMPL  './functions/chat_with_df.py';
            """

cursor.query("DROP FUNCTION IF EXISTS ChatWithPandas;").execute()

cursor.query(create_function_query).execute()
print("Created Function")

create_table_query = f"""CREATE TABLE IF NOT EXISTS AIRBNB_DATA5(
    bathrooms FLOAT(64, 64),
    bedrooms FLOAT(64, 64),
    beds FLOAT(64, 64),
    location_name TEXT(255),
    num_guests FLOAT(64, 64),
    num_reviews FLOAT(64, 64),
    price FLOAT(64, 64),
    rating TEXT(225),
    latitude FLOAT(64, 64),
    longitude FLOAT(64, 64),
    zipcode TEXT(10),
    pop2016 FLOAT(64, 64),
    pop2010 FLOAT(64, 64),
    pop2000 FLOAT(64, 64),
    cost_living_index FLOAT(64, 64),
    land_area FLOAT(64, 64),
    water_area FLOAT(64, 64),
    pop_density INTEGER,
    number_of_males INTEGER,
    number_of_females INTEGER,
    prop_taxes_paid_2016 FLOAT(64, 64),
    median_taxes_with_mortgage FLOAT(64, 64),
    median_taxes_no_mortgage FLOAT(64, 64),
    median_house_value FLOAT(64, 64),
    median_household_income FLOAT(64, 64),
    median_monthly_owner_costs_with_mortgage FLOAT(64, 64),
    median_monthly_owner_costs_no_mortgage FLOAT(64, 64),
    median_gross_rent FLOAT(64, 64),
    median_asking_price_for_sale_home_condo FLOAT(64, 64),
    unemployment FLOAT(64, 64),
    number_of_homes INTEGER,
    count_of_abnb INTEGER,
    density_of_abnb FLOAT(64, 64),
    avg_abnb_price_by_zipcode FLOAT(64, 64),
    avg_num_reviews_by_zipcode FLOAT(64, 64),
    avg_rating_by_zipcode FLOAT(64, 64),
    avg_num_bathrooms_by_zipcode FLOAT(64, 64),
    avg_num_bedrooms_by_zipcode FLOAT(64, 64),
    avg_num_beds_by_zipcode FLOAT(64, 64),
    avg_num_guests_by_zipcode FLOAT(64, 64)
); """

load_data_query = f""" LOAD CSV 'data/Airbnb/missing_values/dirty_test1.csv' INTO AIRBNB_DATA5;"""
cursor.query(create_table_query).df()
cursor.query(load_data_query).df()
print("loaded data")


# data = pd.read_csv('data/Airbnb/missing_values/dirty_test1.csv')
data = pd.read_csv('cleaned_dfs/cleaned_df_int.csv')
# data = pd.read_csv('cleaned_df.csv')

#clean using llm
        # remove duplicate rows.', \

# query = f""" SELECT ChatWithPandas('cleaning',\
#       'impute null values with average of the column if an integer or float. replace with an empty string if column is a string.',\
#             Bathrooms, Bedrooms, Beds, Location_Name, Num_Guests, Num_Reviews, Price, Rating, latitude, longitude, zipcode, pop2016, pop2010, pop2000, cost_living_index, land_area, water_area, pop_density, number_of_males, number_of_females, prop_taxes_paid_2016, median_taxes_with_mortgage, median_taxes_no_mortgage, median_house_value, median_household_income, median_monthly_owner_costs_with_mortgage, median_monthly_owner_costs_no_mortgage, median_gross_rent, median_asking_price_for_sale_home_condo, unemployment, number_of_homes, count_of_abnb, density_of_abnb, avg_abnb_price_by_zipcode, avg_num_reviews_by_zipcode, avg_rating_by_zipcode, avg_num_bathrooms_by_zipcode, avg_num_bedrooms_by_zipcode, avg_num_beds_by_zipcode, avg_num_guests_by_zipcode) FROM AIRBNB_DATA5;
# """
# data = cursor.query(query).execute()
#clean ends here


# data = data.dropna()
# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

data = pd.get_dummies(data, columns=categorical_cols)
# data.dropna()

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Convert to torch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# Define a simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Compute accuracy
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    predictions = (predictions > 0.5).float()
    correct = (predictions == y_test_tensor).float().sum()
    accuracy = correct / len(y_test_tensor)
    print("accuracyy:", accuracy)
    


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"F1 Score: {f1:.2f}")
