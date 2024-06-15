import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, date

# Initialize Faker
fake = Faker()

# Define running shoe models with current versions and release years
current_year = datetime.now().year
shoe_models = {
    'Hoka': {'Bondi': 8, 'Gaviota': 4, 'Arahi': 6, 'Clifton': 9},
    'Nike': {'Pegasus': 39, 'Structure': 24, 'Vomero': 16, 'Invincible': 3},
    'Brooks': {'Ghost': 15, 'Glycerin': 20, 'Adrenaline': 22, 'Glycerin GTS': 22},
    'Asics': {'GT-2000': 11, 'Kayano': 29, 'Cumulus': 24, 'Nimbus': 24},
    'ON': {'Cloudmonster': 1, 'Cloudrunner': 1, 'Cloudflow': 4, 'Cloudsurfer': 6},
    'Saucony': {'Ride': 15, 'Guide': 15, 'Endorphin Speed': 3},
    'Mizuno': {'Waverider': 25, 'Wavesky': 5, 'Waveinspire': 18},
    'New Balance': {'880': 14, '860': 14, '1080': 14}
}

# Define shoe categories and prices
shoe_prices = {
    'Moderate Neutral': 140,
    'High Cushion Neutral': 160,
    'Moderate Stability': 145,
    'High Stability': 165,
}

# Define shoe categories
moderate_neutral_models = ['Clifton', 'Pegasus', 'Cumulus', 'Ghost', 'Cloudsurfer', 'Ride', 'Endorphin Speed', 'Waverider', '880']
high_neutral_models = ['Nimbus', 'Bondi', 'Glycerin', 'Wavesky', '1080', 'Vomero', 'Invincible', 'Cloudmonster']

moderate_stability_models = ['GT-2000', 'Adrenaline', 'Arahi', 'Waveinspire', '860', 'Structure', 'Cloudrunner', 'Guide']
high_stability_models = ['Kayano', 'Glycerin GTS', 'Gaviota', 'Wavehorizon']

# Define customer types and their preferences
customer_types = [
    {
        "type": "Doctor Referral",
        "probability": 0.1,
        "preferred_brands": ["Hoka", "Asics", "New Balance"],
        "insole_probability": 0.5,
        "gender_distribution": [0.5, 0.5],
        "shoe_size_distribution": [10.5, 9],  # Average shoe sizes for male and female
        "discount": "Doctor Referral",
        "credit_card": "Visa",
        "preferred_category": "stability"
    },
    {
        "type": "Student Athlete",
        "probability": 0.1,
        "preferred_brands": ["Nike", "ON", "New Balance"],
        "insole_probability": 0,
        "gender_distribution": [0.6, 0.4],
        "shoe_size_distribution": [10.5, 9],
        "discount": "Student-Athlete",
        "credit_card": "Visa",
        "preferred_category": "neutral"
    },
    {
        "type": "Medical Personnel",
        "probability": 0.15,
        "preferred_brands": ["Hoka", "Asics"],
        "insole_probability": 0.3,
        "gender_distribution": [0.3, 0.7],
        "shoe_size_distribution": [10.5, 9],
        "discount": "Medical Personnel",
        "credit_card": "MasterCard",
        "preferred_category": "stability"
    },
    {
        "type": "Military",
        "probability": 0.05,
        "preferred_brands": ["Brooks", "Mizuno"],
        "insole_probability": 0.2,
        "gender_distribution": [0.7, 0.3],
        "shoe_size_distribution": [10.5, 9],
        "discount": "Military",
        "credit_card": "Cash",
        "preferred_category": "neutral"
    },
    {
        "type": "None",
        "probability": 0.6,
        "preferred_brands": ["All"],
        "insole_probability": 0.2,
        "gender_distribution": [0.5, 0.5],
        "shoe_size_distribution": [10.5, 9],
        "discount": "None",
        "credit_card": "Visa",
        "preferred_category": "neutral"
    }
]

# Ensure the probabilities sum to 1
total_probability = sum(ct['probability'] for ct in customer_types)
if total_probability != 1.0:
    raise ValueError(f"The probabilities of customer types do not sum to 1 (sum: {total_probability})")

# Function to generate customers
def generate_customers(num_customers=1000):
    customers = []
    for _ in range(num_customers):
        customer_id = fake.unique.uuid4()
        name = fake.name()
        email = fake.email()
        phone = fake.phone_number()
        address = fake.address()
        loyalty_points = random.randint(0, 500)
        customer_type = np.random.choice(customer_types, p=[ct['probability'] for ct in customer_types])
        discount = customer_type['discount']
        insole = np.random.choice([True, False], p=[customer_type['insole_probability'], 1 - customer_type['insole_probability']])
        mailing_list = np.random.choice([True, False], p=[0.1, 0.9])
        gender = np.random.choice(['Male', 'Female'], p=customer_type['gender_distribution'])
        shoe_size = customer_type['shoe_size_distribution'][0] if gender == 'Male' else customer_type['shoe_size_distribution'][1]
        credit_card = customer_type['credit_card']
        preferred_category = customer_type['preferred_category']
        customers.append([customer_id, name, email, phone, address, loyalty_points, discount, insole, mailing_list, gender, shoe_size, credit_card, preferred_category])
    return pd.DataFrame(customers, columns=['customer_id', 'name', 'email', 'phone', 'address', 'loyalty_points', 'discount', 'insole', 'mailing_list', 'gender', 'shoe_size', 'credit_card', 'preferred_category'])

# Function to generate purchase history
def generate_purchase_history(customers, min_purchases=3, max_purchases=6):
    purchases = []
    for _, customer in customers.iterrows():
        num_purchases = random.randint(min_purchases, max_purchases)
        for _ in range(num_purchases):
            purchase_year = random.randint(current_year - 4, current_year)
            start_date = date(purchase_year, 1, 1)
            end_date = date(purchase_year, 12, 31)
            purchase_date = fake.date_between(start_date=start_date, end_date=end_date)
            
            # Determine the brand based on customer type preferences
            if customer['discount'] != 'None':
                preferred_brands = [ct['preferred_brands'] for ct in customer_types if ct['discount'] == customer['discount']][0]
                if 'All' in preferred_brands:
                    brand = np.random.choice(list(shoe_models.keys()))
                else:
                    brand = np.random.choice(preferred_brands)
            else:
                brand = np.random.choice(list(shoe_models.keys()))
                
            model_name, current_version = random.choice(list(shoe_models[brand].items()))
            model_version = max(current_version - (current_year - purchase_year), 1)  # Ensure model version is at least 1
            model = f"{model_name} {model_version}"
            
            if customer['preferred_category'] == 'neutral':
                if model_name in moderate_neutral_models:
                    category = 'Moderate Neutral'
                else:
                    category = 'High Cushion Neutral'
            else:
                if model_name in moderate_stability_models:
                    category = 'Moderate Stability'
                else:
                    category = 'High Stability'
                    
            price = shoe_prices[category]
            transaction_type = 'purchase'
            
            if random.random() < 0.1:  # 10% chance to return
                return_date = fake.date_between(start_date=purchase_date, end_date=end_date)
                return_price = -price
                purchases.append([customer['customer_id'], brand, model, return_date, return_price, 'return'])
                
            purchases.append([customer['customer_id'], brand, model, purchase_date, price, transaction_type])
            
    return pd.DataFrame(purchases, columns=['customer_id', 'brand', 'model', 'purchase_date', 'price', 'transaction_type'])

# Generate synthetic customer data
customers_df = generate_customers(num_customers=1000)
purchase_history_df = generate_purchase_history(customers_df)

# Save to CSV files
customers_df.to_csv('customers.csv', index=False)
purchase_history_df.to_csv('purchase_history.csv', index=False)

# Analysis functions
def analyze_customers(customers_df):
    print("Customer Data Analysis")
    print("======================")
    print(customers_df.describe(include='all'))

def analyze_purchase_history(purchase_history_df):
    print("\nPurchase History Analysis")
    print("=========================")
    print(purchase_history_df.describe(include='all'))

def print_model_distribution_by_discount(purchase_history_df, customers_df):
    merged_df = pd.merge(purchase_history_df, customers_df[['customer_id', 'discount']], on='customer_id')
    discount_groups = merged_df.groupby('discount')
    
    for discount, group in discount_groups:
        model_counts = group['model'].value_counts(normalize=True) * 100
        print(f"\nModel Distribution for {discount} (by %):")
        print(model_counts)

def print_model_distribution_by_size(purchase_history_df, customers_df):
    merged_df = pd.merge(purchase_history_df, customers_df[['customer_id', 'shoe_size']], on='customer_id')
    size_groups = merged_df.groupby('shoe_size')
    
    for size, group in size_groups:
        model_counts = group['model'].value_counts(normalize=True) * 100
        print(f"\nModel Distribution for Shoe Size {size} (by %):")
        print(model_counts)

def print_model_distribution_by_insole(purchase_history_df, customers_df):
    merged_df = pd.merge(purchase_history_df, customers_df[['customer_id', 'insole']], on='customer_id')
    insole_groups = merged_df.groupby('insole')
    
    for insole, group in insole_groups:
        model_counts = group['model'].value_counts(normalize=True) * 100
        print(f"\nModel Distribution for Insole Usage {insole} (by %):")
        print(model_counts)

def check_adherence_to_rules(customers_df, purchase_history_df, customer_types, moderate_neutral_models, high_neutral_models, moderate_stability_models, high_stability_models):
    print("\nAdherence to Rules")
    print("===================")

    # Filter out rows with NaN values in the 'model' column
    purchase_history_df = purchase_history_df.dropna(subset=['model'])

    # Rule: 80% chance to buy from preferred brand
    preferred_brands_count = 0
    total_brands_checked = 0
    for _, row in customers_df.iterrows():
        customer_purchases = purchase_history_df[purchase_history_df['customer_id'] == row['customer_id']]
        for _, purchase in customer_purchases.iterrows():
            if purchase['brand'] in [ct['preferred_brands'] for ct in customer_types if ct['type'] == row['discount']][0]:
                preferred_brands_count += 1
            total_brands_checked += 1
    preferred_brands_ratio = preferred_brands_count / total_brands_checked
    print(f"Preferred Brand Purchase Ratio: {preferred_brands_ratio:.2f} (Expected: 0.80)")

    # Rule: 80% chance to stick to their category (neutral/stability)
    preferred_category_count = 0
    total_categories_checked = 0
    for _, row in customers_df.iterrows():
        customer_purchases = purchase_history_df[purchase_history_df['customer_id'] == row['customer_id']]
        for _, purchase in customer_purchases.iterrows():
            model_name = purchase['model'].split()[0]
            if row['preferred_category'] == 'neutral' and model_name in moderate_neutral_models + high_neutral_models:
                preferred_category_count += 1
            elif row['preferred_category'] == 'stability' and model_name in moderate_stability_models + high_stability_models:
                preferred_category_count += 1
            total_categories_checked += 1
    preferred_category_ratio = preferred_category_count / total_categories_checked
    print(f"Preferred Category Purchase Ratio: {preferred_category_ratio:.2f} (Expected: 0.80)")

# Run the analysis
analyze_customers(customers_df)
analyze_purchase_history(purchase_history_df)
print_model_distribution_by_discount(purchase_history_df, customers_df)
print_model_distribution_by_size(purchase_history_df, customers_df)
print_model_distribution_by_insole(purchase_history_df, customers_df)
check_adherence_to_rules(customers_df, purchase_history_df, customer_types, moderate_neutral_models, high_neutral_models, moderate_stability_models, high_stability_models)
