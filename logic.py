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

# read csv files
try:
    customers_df = pd.read_csv('data/customers.csv')
    purchase_history_df = pd.read_csv('data/purchase_history.csv')
except Exception as e:
    print(f"Error: {e}")
    exit()

# generate a list of all unique models in purchase history
def get_all_models_list():
    return purchase_history_df['model'].unique().tolist()

# generate a list of all unique customers
def get_all_customers_list():
    return customers_df['name'].unique().tolist()

# function to get purchases for a customer
def purchases(customer_id, history_df):
    return history_df[history_df['customer_id'] == customer_id][['model', 'purchase_date', 'price', 'transaction_type']].values.tolist()

# function to check if a customer has made returns
def returns(purchases):
    return any('return' in instance for instance in purchases)

# function to get customer id by name or phone number
def get_id(input, customers):
    if input.isdigit():
        for index, row in customers.iterrows():
            if row['phone'] == input:
                return row['customer_id']
    else:
        for index, row in customers.iterrows():
            if row['name'] == input:
                return row['customer_id']
    return None

# function to calculate the average time between purchases
def time_between_purchases(purchases):
    dates = [instance[1] for instance in purchases]
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    date_objects.sort()
    gaps = [(date_objects[i] - date_objects[i-1]).days for i in range(1, len(date_objects))]
    average_gap = np.mean(gaps) if gaps else np.nan
    return average_gap

# function to get customer data
def get_customer_data(customer_id, customers_df, purchase_history_df):
    """returns a dictionary of customer data, without recommendations"""
    customer_info = customers_df[customers_df['customer_id'] == customer_id]
    if customer_info.empty:
        return None
    customer_info = customer_info.iloc[0].to_dict()
    purchase_history = purchases(customer_id, purchase_history_df)
    customer_info['purchase_history'] = purchase_history
    customer_info['returner'] = returns(purchase_history)
    customer_info['avg_time_between_purchases'] = time_between_purchases(purchase_history)
    return customer_info

# add returner and average time between purchases to dataframe
customers_df['returner'] = customers_df['customer_id'].apply(lambda x: returns(purchases(x, purchase_history_df)))
customers_df['avg_time_between_purchases'] = customers_df['customer_id'].apply(lambda x: time_between_purchases(purchases(x, purchase_history_df)))
customers_df['avg_time_between_purchases'].fillna(customers_df['avg_time_between_purchases'].mean(), inplace=True)

# create interaction matrix
interaction_matrix = purchase_history_df.pivot_table(index='customer_id', columns='model', values='price', aggfunc='count', fill_value=0)

# extract additional features
additional_features = customers_df[['shoe_size', 'gender', 'preferred_category', 'discount', 'insole', 'mailing_list', 'returner', 'avg_time_between_purchases']]

# one-hot encode categorical features and standardize numerical features
column_transformer = ColumnTransformer([
    ('gender', OneHotEncoder(), ['gender']),
    ('preferred_category', OneHotEncoder(), ['preferred_category']),
    ('discount', OneHotEncoder(), ['discount']),
    ('insole', OneHotEncoder(), ['insole']),
    ('mailing_list', OneHotEncoder(), ['mailing_list']),
    ('returner', OneHotEncoder(), ['returner'])
], remainder='passthrough')

pipeline = Pipeline([
    ('onehot', column_transformer),
    ('scaler', StandardScaler())
])

additional_features_encoded = pipeline.fit_transform(additional_features)

# apply matrix factorization to the interaction matrix
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(interaction_matrix)
item_factors = svd.components_.T

# combine latent factors with additional features for users
combined_user_features = np.hstack([user_factors, additional_features_encoded])
dummy_additional_features = np.zeros((item_factors.shape[0], additional_features_encoded.shape[1]))
combined_item_features = np.hstack([item_factors, dummy_additional_features])

# fit nearest neighbors model on the combined user features
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(combined_user_features)

# predict function
def predict(user_idx, item_idx):
    """returns the 'predicted score' for a model based on user_idx and item_idx"""
    user_vector = combined_user_features[user_idx]
    item_vector = combined_item_features[item_idx]
    return np.dot(user_vector, item_vector)

# rank items for a specific user
def rank_items_for_user(user_id, interaction_matrix):
    """ranks all models in unique models by score for user"""
    user_idx = interaction_matrix.index.get_loc(user_id)
    item_scores = [(interaction_matrix.columns[item_idx], predict(user_idx, item_idx)) for item_idx in range(len(interaction_matrix.columns))]
    ranked_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
    return ranked_items

# get user data and recommendations
def get(name):
    """returns all user data plus recommendations"""
    user_id = get_id(input=name, customers=customers_df)
    if not user_id:
        return None
    data = get_customer_data(user_id, customers_df, purchase_history_df)
    ranked_items = rank_items_for_user(user_id, interaction_matrix)
    data['recommendations'] = ranked_items
    return data

# postprocess data to csv
def postprocess(data, csv_file):
    purchase_history_rows = [{'customer_id': data.get('customer_id', ''), 'name': data.get('name', ''), 'item': item[0], 'date': item[1], 'amount': item[2], 'type': item[3]} for item in data.get('purchase_history', [])]
    recommendation_rows = [{'customer_id': data.get('customer_id', ''), 'name': data.get('name', ''), 'item': rec[0], 'score': rec[1], 'type': 'recommendation'} for rec in data.get('recommendations', [])]
    combined_df = pd.concat([pd.DataFrame(purchase_history_rows), pd.DataFrame(recommendation_rows)], ignore_index=True)
    file_exists = os.path.isfile(csv_file)
    combined_df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

# rank users by item
def rank_users_by_item(item_idx, customers_df, interaction_matrix):
    user_scores = [(customers_df.iloc[user_idx]['customer_id'], customers_df.iloc[user_idx]['name'], customers_df.iloc[user_idx]['phone'], customers_df.iloc[user_idx]['email'], predict(user_idx=user_idx, item_idx=item_idx)) for user_idx in range(customers_df.shape[0])]
    ranked_users = sorted(user_scores, key=lambda x: x[4], reverse=True)
    return ranked_users

# get user rankings for an item
def get_user_rankings(item_name, amount_of_users):
    """returns a list of length [amount_of_users] where list[0] = customer_id, list[1] = customer_name, list[2] = item_score"""
    shoe_idx = get_shoe_idx(item_name)
    ranking_list = rank_users_by_item(shoe_idx, customers_df, interaction_matrix)
    return ranking_list[0:amount_of_users]

# get shoe index
def get_shoe_idx(shoe_name):
    """returns the index of the shoe in the interaction matrix columns"""
    return interaction_matrix.columns.get_loc(shoe_name)
    
def filter(data_list, gender, size, discount):
    filtered_customers = data_list

    if gender and gender != 'any':
        filtered_customers = [person for person in filtered_customers if get_customer_data(person[0], customers_df, purchase_history_df)['gender'] == gender]
    
    if size is not None and size != 'any':
        try:
            size = float(size)
            filtered_customers = [person for person in filtered_customers if get_customer_data(person[0], customers_df, purchase_history_df)['shoe_size'] == size]
        except ValueError:
            pass  # Handle any conversion error or inappropriate size input
    
    if discount and discount != 'any':
        filtered_customers = [person for person in filtered_customers if get_customer_data(person[0], customers_df, purchase_history_df)['discount'] == discount]
    
    return filtered_customers
