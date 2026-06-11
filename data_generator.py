import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

os.makedirs('data', exist_ok=True)

first_names = [
    'Ava', 'Liam', 'Noah', 'Emma', 'Olivia', 'Elijah', 'Charlotte', 'Amelia', 'Sophia', 'Mason',
    'Lucas', 'Logan', 'Mia', 'Ethan', 'Harper', 'James', 'Evelyn', 'Jackson', 'Abigail', 'Aiden',
    'Carter', 'Avery', 'Henry', 'Scarlett', 'Sebastian', 'Grace', 'Owen', 'Chloe', 'Wyatt', 'Ella',
    'Leo', 'Luna', 'Jacob', 'Aria', 'Samuel', 'Nora', 'Matthew', 'Lily', 'Isaac', 'Hannah',
    'Jack', 'Zoe', 'Julian', 'Riley', 'Ryan', 'Victoria', 'Nathan', 'Stella', 'Dylan', 'Penelope'
]

last_names = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia', 'Rodriguez', 'Wilson',
    'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White',
    'Lopez', 'Lee', 'Gonzalez', 'Harris', 'Clark', 'Lewis', 'Robinson', 'Walker', 'Perez', 'Hall'
]

product_names = [
    'Smartphone', 'Laptop', 'Wireless Earbuds', 'Fitness Tracker', 'Smartwatch', 'Tablet',
    'Noise Cancelling Headphones', 'Gaming Console', 'Electric Scooter', '4K Monitor',
    'Portable Charger', 'Coffee Maker', 'Blender', 'Air Fryer', 'Premium Desk Chair',
    'Office Lamp', 'Backpack', 'Running Shoes', 'Jacket', 'Designer Sunglasses',
    'Yoga Mat', 'Water Bottle', 'Bluetooth Speaker', 'Digital Camera', 'Streaming Device'
]

categories = ['Electronics', 'Home', 'Fitness', 'Accessories', 'Outdoor']

customers = []
for i in range(1, 121):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = int(np.clip(np.random.normal(36, 10), 18, 65))
    salary = int(np.clip(np.random.normal(63000, 18000), 22000, 140000))
    expenditures = int(np.clip(np.random.normal(salary * 0.55, salary * 0.18), salary * 0.2, salary * 0.95))
    customers.append({
        'customer_id': i,
        'name': name,
        'age': age,
        'salary': salary,
        'monthly_expenditures': expenditures
    })

customer_df = pd.DataFrame(customers)
customer_df.to_csv('data/customers.csv', index=False)

products = []
for i, product in enumerate(product_names, start=1):
    category = random.choice(categories)
    base_price = int(np.clip(np.random.normal(120 + i * 5, 35), 20, 650))
    min_salary = int(np.clip(base_price * random.uniform(2.0, 5.5), 28000, 130000))
    products.append({
        'product_id': i,
        'product_name': product,
        'category': category,
        'price': base_price,
        'recommended_min_salary': min_salary
    })

product_df = pd.DataFrame(products)
product_df.to_csv('data/products.csv', index=False)

transactions = []
for row in customers:
    monthly_trans = int(np.clip(np.random.poisson(12) + (row['monthly_expenditures'] // 1500), 2, 45))
    churn = int((monthly_trans < 5) or (row['monthly_expenditures'] > row['salary'] * 0.9))
    transactions.append({
        'customer_id': row['customer_id'],
        'monthly_transactions': monthly_trans,
        'churn': churn,
        'active_months': int(np.clip(np.random.normal(10, 3), 3, 12))
    })

transaction_df = pd.DataFrame(transactions)
transaction_df.to_csv('data/transactions.csv', index=False)

print('Generated data/customers.csv, data/products.csv, data/transactions.csv')
