import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Constants for realistic data generation
MERCHANT_CATEGORIES = [
    'grocery', 'gas_station', 'electronics', 'restaurant', 'online_retail',
    'apparel', 'entertainment', 'travel', 'health', 'home_improvement',
    'pharmacy', 'education', 'utilities', 'subscriptions', 'transportation',
    'hotels', 'beauty', 'gifts', 'automotive', 'sports'
]

CHANNELS = ['pos', 'online', 'mobile_app', 'atm']

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def generate_customer_profiles(n_customers: int, seed: int = 42) -> dict:
    np.random.seed(seed)
    profiles = {}
    for i in range(n_customers):
        customer_id = f"CUST_{str(uuid.uuid4())[:8]}"
        profiles[customer_id] = {
            'typical_amount_mean': np.random.uniform(20.0, 150.0),
            'typical_amount_std': np.random.uniform(5.0, 30.0),
            'preferred_merchant_categories': np.random.choice(MERCHANT_CATEGORIES, size=np.random.randint(3, 6), replace=False).tolist(),
            'preferred_hours': np.random.choice(range(24), size=np.random.randint(3, 6), replace=False).tolist(),
            'home_location': (np.random.uniform(-90, 90), np.random.uniform(-180, 180)),
            'primary_device_id': f"DEV_{str(uuid.uuid4())[:8]}",
            'transaction_frequency': np.random.uniform(0.5, 3.0) # avg per day
        }
    return profiles

def velocity_abuse(customer_id, profile, timestamp):
    """burst of 5-15 transactions in a 5-minute window, normal amounts."""
    n_tx = np.random.randint(5, 16)
    txs = []
    base_ts = timestamp
    for _ in range(n_tx):
        amt = max(1.0, np.random.normal(profile['typical_amount_mean'], profile['typical_amount_std']))
        txs.append({
            'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
            'customer_id': customer_id,
            'amount': round(amt, 2),
            'merchant_id': f"M_{np.random.randint(1000, 9999)}",
            'merchant_category': np.random.choice(profile['preferred_merchant_categories']),
            'device_id': profile['primary_device_id'],
            'ip_address': f"192.168.1.{np.random.randint(1, 255)}",
            'location_lat': profile['home_location'][0] + np.random.normal(0, 0.01),
            'location_lon': profile['home_location'][1] + np.random.normal(0, 0.01),
            'timestamp': base_ts + timedelta(seconds=np.random.randint(0, 300)),
            'currency': 'USD',
            'channel': np.random.choice(CHANNELS),
            'is_fraud': 1,
            'attack_family': 'velocity_abuse'
        })
    return txs

def card_testing(customer_id, profile, timestamp):
    """10-30 small transactions ($0.50-$5.00) across many different merchants within an hour."""
    n_tx = np.random.randint(10, 31)
    txs = []
    base_ts = timestamp
    for _ in range(n_tx):
        amt = np.random.uniform(0.50, 5.00)
        txs.append({
            'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
            'customer_id': customer_id,
            'amount': round(amt, 2),
            'merchant_id': f"M_{np.random.randint(1000, 9999)}",
            'merchant_category': np.random.choice(MERCHANT_CATEGORIES),
            'device_id': profile['primary_device_id'],
            'ip_address': f"10.0.0.{np.random.randint(1, 255)}",
            'location_lat': profile['home_location'][0] + np.random.normal(0, 0.05),
            'location_lon': profile['home_location'][1] + np.random.normal(0, 0.05),
            'timestamp': base_ts + timedelta(seconds=np.random.randint(0, 3600)),
            'currency': 'USD',
            'channel': 'online',
            'is_fraud': 1,
            'attack_family': 'card_testing'
        })
    return txs

def account_takeover(customer_id, profile, timestamp):
    """sudden device change + location change + 1-3 high-value transactions (3-10x)."""
    n_tx = np.random.randint(1, 4)
    txs = []
    base_ts = timestamp
    new_device = f"DEV_ATO_{str(uuid.uuid4())[:5]}"
    new_lat = np.random.uniform(-90, 90)
    new_lon = np.random.uniform(-180, 180)
    
    for _ in range(n_tx):
        amt = profile['typical_amount_mean'] * np.random.uniform(3.0, 10.0)
        txs.append({
            'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
            'customer_id': customer_id,
            'amount': round(amt, 2),
            'merchant_id': f"M_{np.random.randint(1000, 9999)}",
            'merchant_category': 'electronics',
            'device_id': new_device,
            'ip_address': f"203.0.113.{np.random.randint(1, 255)}",
            'location_lat': new_lat,
            'location_lon': new_lon,
            'timestamp': base_ts + timedelta(seconds=np.random.randint(0, 600)),
            'currency': 'USD',
            'channel': 'online',
            'is_fraud': 1,
            'attack_family': 'account_takeover'
        })
    return txs

def geo_impossible_travel(customer_id, profile, timestamp):
    """two transactions in locations >1000km apart within 30 minutes."""
    txs = []
    base_ts = timestamp
    lat1, lon1 = profile['home_location']
    
    # First legit-looking tx
    amt1 = max(1.0, np.random.normal(profile['typical_amount_mean'], profile['typical_amount_std']))
    txs.append({
        'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
        'customer_id': customer_id,
        'amount': round(amt1, 2),
        'merchant_id': f"M_{np.random.randint(1000, 9999)}",
        'merchant_category': np.random.choice(profile['preferred_merchant_categories']),
        'device_id': profile['primary_device_id'],
        'ip_address': f"192.168.1.{np.random.randint(1, 255)}",
        'location_lat': lat1,
        'location_lon': lon1,
        'timestamp': base_ts,
        'currency': 'USD',
        'channel': 'pos',
        'is_fraud': 1,
        'attack_family': 'geo_impossible_travel'
    })
    
    # Second impossible tx
    lat2 = lat1 + 20.0 if lat1 < 70 else lat1 - 20.0
    lon2 = lon1 + 20.0 if lon1 < 160 else lon1 - 20.0
    amt2 = max(1.0, np.random.normal(profile['typical_amount_mean'], profile['typical_amount_std']))
    
    txs.append({
        'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
        'customer_id': customer_id,
        'amount': round(amt2, 2),
        'merchant_id': f"M_{np.random.randint(1000, 9999)}",
        'merchant_category': np.random.choice(MERCHANT_CATEGORIES),
        'device_id': profile['primary_device_id'],
        'ip_address': f"198.51.100.{np.random.randint(1, 255)}",
        'location_lat': lat2,
        'location_lon': lon2,
        'timestamp': base_ts + timedelta(minutes=np.random.randint(5, 25)),
        'currency': 'USD',
        'channel': 'pos',
        'is_fraud': 1,
        'attack_family': 'geo_impossible_travel'
    })
    return txs

def synthetic_identity(timestamp):
    """creates a brand-new customer_id with NO prior history, immediately making 2-5 high-value transactions."""
    new_customer = f"CUST_SYN_{str(uuid.uuid4())[:8]}"
    n_tx = np.random.randint(2, 6)
    txs = []
    base_ts = timestamp
    
    device_id = f"DEV_SYN_{str(uuid.uuid4())[:8]}"
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)
    
    for _ in range(n_tx):
        amt = np.random.uniform(500.0, 5000.0)
        txs.append({
            'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
            'customer_id': new_customer,
            'amount': round(amt, 2),
            'merchant_id': f"M_{np.random.randint(1000, 9999)}",
            'merchant_category': np.random.choice(['electronics', 'jewelry', 'cash_advance']),
            'device_id': device_id,
            'ip_address': f"172.16.0.{np.random.randint(1, 255)}",
            'location_lat': lat,
            'location_lon': lon,
            'timestamp': base_ts + timedelta(seconds=np.random.randint(0, 86400)),
            'currency': 'USD',
            'channel': 'online',
            'is_fraud': 1,
            'attack_family': 'synthetic_identity'
        })
    return txs

def generate_dataset(n_customers=500, n_legit_per_customer=50, fraud_ratio=0.05, attack_families=None, seed=42):
    np.random.seed(seed)
    profiles = generate_customer_profiles(n_customers, seed)
    
    start_date = datetime.now() - timedelta(days=30)
    transactions = []
    
    all_attacks = ['velocity_abuse', 'card_testing', 'account_takeover', 'geo_impossible_travel', 'synthetic_identity']
    if attack_families is None:
        attack_families = all_attacks
    
    total_target = n_customers * n_legit_per_customer
    fraud_target = int(total_target * fraud_ratio)
    
    for c_id, profile in profiles.items():
        # Legit txs
        for _ in range(n_legit_per_customer):
            day_offset = np.random.randint(0, 30)
            hour = np.random.choice(profile['preferred_hours'])
            minute = np.random.randint(0, 60)
            ts = start_date + timedelta(days=day_offset, hours=int(hour), minutes=minute)
            
            amt = max(0.5, np.random.normal(profile['typical_amount_mean'], profile['typical_amount_std']))
            transactions.append({
                'transaction_id': f"TX_{str(uuid.uuid4())[:8]}",
                'customer_id': c_id,
                'amount': round(amt, 2),
                'merchant_id': f"M_{np.random.randint(1000, 9999)}",
                'merchant_category': np.random.choice(profile['preferred_merchant_categories']),
                'device_id': profile['primary_device_id'],
                'ip_address': f"192.168.1.{np.random.randint(1, 255)}",
                'location_lat': profile['home_location'][0] + np.random.normal(0, 0.02),
                'location_lon': profile['home_location'][1] + np.random.normal(0, 0.02),
                'timestamp': ts,
                'currency': 'USD',
                'channel': np.random.choice(CHANNELS),
                'is_fraud': 0,
                'attack_family': 'legitimate'
            })
            
    # Add fraud
    fraud_count = 0
    while fraud_count < fraud_target:
        c_id = np.random.choice(list(profiles.keys()))
        profile = profiles[c_id]
        
        day_offset = np.random.randint(0, 30)
        hour = np.random.randint(0, 24)
        ts = start_date + timedelta(days=day_offset, hours=hour)
        
        attack = np.random.choice(attack_families)
        
        if attack == 'velocity_abuse':
            txs = velocity_abuse(c_id, profile, ts)
        elif attack == 'card_testing':
            txs = card_testing(c_id, profile, ts)
        elif attack == 'account_takeover':
            txs = account_takeover(c_id, profile, ts)
        elif attack == 'geo_impossible_travel':
            txs = geo_impossible_travel(c_id, profile, ts)
        elif attack == 'synthetic_identity':
            txs = synthetic_identity(ts)
            
        transactions.extend(txs)
        fraud_count += len(txs)
        
    df = pd.DataFrame(transactions)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

if __name__ == '__main__':
    print("Generating synthetic dataset...")
    df = generate_dataset(n_customers=500, n_legit_per_customer=50, fraud_ratio=0.05, seed=42)
    
    print("\n=== Dataset Summary ===")
    print(f"Total Transactions: {len(df)}")
    print(f"Total Fraudulent: {df['is_fraud'].sum()}")
    print(f"Fraud Rate: {df['is_fraud'].mean():.2%}")
    print("\nAttack Families Breakdown:")
    print(df['attack_family'].value_counts())
    
    out_dir = Path(__file__).parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_transactions.csv"
    
    df.to_csv(out_path, index=False)
    print(f"\nSaved synthetic dataset to: {out_path}")
