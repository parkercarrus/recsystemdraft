# logic.py
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import os

customers_df = pd.read_csv('path_to_customer_data.csv') # replace this with the actual path to csv --> should be /data/customers.csv
purchase_history_df = pd.read_csv('path_to_purchase_history.csv') # same idea

# generate a list of all unique models in purchase history
def get_all_models_list():
    """Returns a list of all the unique models in purchase history"""
    models = []
    for model in purchase_history_df['model']:
        if model not in models:
            models.append(model)
    return models

# Function to get purchases for customer
def purchases(customer_id, history_df):
    return history_df[history_df['customer_id'] == customer_id][['model', 'purchase_date', 'price', 'transaction_type']].values.tolist()

# Function to check if a customer has made returns
def returns(purchases):
    return any('return' in instance for instance in purchases)

def get_id(input, customers):
    if input.isdigit():
        for index, row in customers.iterrows():
            if row['phone'] == input:
                return row['customer_id']
    else:
        for index, row in customers.iterrows():
            if row['name'] == input:
                return row['customer_id']

def time_between_purchases(purchases):
    dates = []
    for instance in enumerate(purchases):
        dates.append(instance[1][1])
    # Convert strings to datetime objects
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    # Sort the dates
    date_objects.sort()
    # Calculate the gaps between dates
    gaps = [(date_objects[i] - date_objects[i-1]).days for i in range(1, len(date_objects))]
    # Calculate the average gap
    average_gap = np.mean(gaps)

    return average_gap
    
        
# Function to get customer data
def get_customer_data(customer_id, customers_df, purchase_history_df):
    """Returns a dictionary of customer data, without recommendations"""
    customer_info = customers_df[customers_df['customer_id'] == customer_id]
    if customer_info.empty:
        return None
    customer_info = customer_info.iloc[0].to_dict()
    purchase_history = purchases(customer_id, purchase_history_df)
    customer_info['purchase_history'] = purchase_history
    customer_info['returner'] = returns(purchase_history)
    customer_info['avg_time_between_purchases'] = time_between_purchases(purchase_history)

    return customer_info

# returner and time between purchases to dataframe
customers_df['returner'] = customers_df['customer_id'].apply(lambda x: returns(purchases(x, purchase_history_df)))
customers_df['avg_time_between_purchases'] = customers_df['customer_id'].apply(lambda x: time_between_purchases(purchases(x, purchase_history_df)))

# fill NAN 'avg_time' values with mean
customers_df['avg_time_between_purchases'].fillna(customers_df['avg_time_between_purchases'].mean(), inplace=True)

# Create interaction matrix
interaction_matrix = purchase_history_df.pivot_table(index='customer_id', columns='model', values='price', aggfunc='count', fill_value=0)

# Extract additional features
additional_features = customers_df[['shoe_size', 'gender', 'preferred_category', 'discount', 'insole', 'mailing_list', 'returner', 'avg_time_between_purchases']]

# One-hot encode categorical features
column_transformer = ColumnTransformer([
    ('gender', OneHotEncoder(), ['gender']),
    ('preferred_category', OneHotEncoder(), ['preferred_category']),
    ('discount', OneHotEncoder(), ['discount']),
    ('insole', OneHotEncoder(), ['insole']),
    ('mailing_list', OneHotEncoder(), ['mailing_list']),
    ('returner', OneHotEncoder(), ['returner'])
], remainder='passthrough')

# Standardize numerical features
pipeline = Pipeline([
    ('onehot', column_transformer),
    ('scaler', StandardScaler())
])

additional_features_encoded = pipeline.fit_transform(additional_features)

# Apply matrix factorization to the interaction matrix
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(interaction_matrix)
item_factors = svd.components_.T

# Combine latent factors with additional features for users
combined_user_features = np.hstack([user_factors, additional_features_encoded])

# dummy additional features to ensure equal matrix size
dummy_additional_features = np.zeros((item_factors.shape[0], additional_features_encoded.shape[1]))

# Combine item factors with dummy additional features
combined_item_features = np.hstack([item_factors, dummy_additional_features])

# Fit Nearest Neighbors model on the combined user features
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(combined_user_features)

# Predict function
def predict(user_idx, item_idx):
    """Returns the 'predicted score' for a model based on user_idx and item_idx"""
    user_vector = combined_user_features[user_idx]
    item_vector = combined_item_features[item_idx]
    return np.dot(user_vector, item_vector)

# Rank items for a specific user
def rank_items_for_user(user_id, interaction_matrix):
    """Ranks all models in UNIQUE MODELS by score for user"""
    user_idx = interaction_matrix.index.get_loc(user_id)
    item_scores = []

    for item_idx in range(len(interaction_matrix.columns)):
        score = predict(user_idx, item_idx)
        item_scores.append((interaction_matrix.columns[item_idx], score))

    # Sort items by score in descending order
    ranked_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
    return ranked_items


def get(name):
    """Returns all all user data plus recommendations"""
    user_id = get_id(input=name, customers=customers_df)
    data = get_customer_data(user_id, customers_df, purchase_history_df)
    ranked_items = rank_items_for_user(user_id, interaction_matrix)
    data['recommendations'] = ranked_items

    return data

def postprocess(data, csv_file):
    # Initialize lists to hold the rows for purchase history and recommendations
    purchase_history_rows = []
    recommendation_rows = []
    
    # Extract customer_id and name
    customer_id = data.get('customer_id', '')
    name = data.get('name', '')
    
    # Extract purchase history
    purchase_history = data.get('purchase_history', [])
    for item in purchase_history:
        purchase_history_rows.append({
            'customer_id': customer_id,
            'name': name,
            'item': item[0],
            'date': item[1],
            'amount': item[2],
            'type': item[3],
        })
    
    # Extract recommendations
    recommendations = data.get('recommendations', [])
    for rec in recommendations:
        recommendation_rows.append({
            'customer_id': customer_id,
            'name': name,
            'item': rec[0],
            'score': rec[1],
            'type': 'recommendation'
        })
    
    # convert lists to dataframes
    purchase_history_df = pd.DataFrame(purchase_history_rows)
    recommendations_df = pd.DataFrame(recommendation_rows)
    
    # combine DataFrames
    combined_df = pd.concat([purchase_history_df, recommendations_df], ignore_index=True)
    
    # Save to CSV
    file_exists = os.path.isfile(csv_file)
    combined_df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

def rank_users_by_item(item_idx, customers_df, interaction_matrix):
    user_scores = []

    # Iterate through each user
    for user_idx in range(customers_df.shape[0]):
        score = predict(user_idx=user_idx, item_idx=item_idx)
        user_scores.append((customers_df.iloc[user_idx]['customer_id'], customers_df.iloc[user_idx]['name'], customers_df.iloc[user_idx]['phone'], customers_df.iloc[user_idx]['email'], score))
        
    # Sort users based on the score in descending order
    ranked_users = sorted(user_scores, key=lambda x: x[4], reverse=True)
    
    return ranked_users

def get_user_rankings(item_name, amount_of_users):
    """Returns a list of length [amount_of_users] where list[0] = customer_id, list[1] = customer_name, list[2] = item_score"""
    shoe_idx = get_shoe_idx(item_name)
    ranking_list = rank_users_by_item(shoe_idx, customers_df, interaction_matrix)
    return ranking_list[0:amount_of_users]
