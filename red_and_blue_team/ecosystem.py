"""
Synthetic payment ecosystem generator.

Produces a small financial world: customers, merchants, accounts, devices,
and the transactions that connect them over time. Each customer has a
spending profile (favorite categories, typical amount, active hours, home
city, travel tendency) that drives their transaction history, so entities
have life histories rather than being independent rows.

This is the base layer only - no fraud is injected here. A red-team agent
can later act on top of this ecosystem.
"""

import csv
import json
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

SEED = 42

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Pune", "Hyderabad", "Jaipur"]

# approximate city-center coordinates, for R0 schema compatibility with Blue's
# location_lat/location_lon fields
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
}

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "dining", "fuel", "travel",
    "clothing", "pharmacy", "entertainment", "utilities", "online_marketplace",
]

DEVICE_TYPES = ["mobile_app", "web_browser", "pos_terminal", "atm"]

# maps our device types to Blue's exact channel values (data/generate_synthetic.py CHANNELS)
CHANNEL_MAP = {
    "mobile_app": "mobile_app",
    "web_browser": "online",
    "pos_terminal": "pos",
    "atm": "atm",
}
CURRENCY = "USD"  # matches Blue's synthetic generator, which is fixed to USD

ACCOUNT_TYPES = ["checking", "savings", "credit_card"]


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class Device:
    device_id: str
    device_type: str
    first_seen: str


@dataclass
class Account:
    account_id: str
    customer_id: str
    account_type: str
    opened_at: str
    device_ids: list = field(default_factory=list)


@dataclass
class Customer:
    customer_id: str
    name: str
    home_city: str
    joined_at: str
    favorite_categories: list
    avg_amount: float
    amount_std: float
    daily_txn_rate: float
    active_hours: tuple
    location_weights: dict
    account_ids: list = field(default_factory=list)


@dataclass
class Merchant:
    merchant_id: str
    name: str
    category: str
    city: str


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    account_id: str
    device_id: str
    merchant_id: str
    amount: float
    timestamp: str
    city: str
    merchant_category: str
    location_lat: float
    location_lon: float
    ip_address: str
    currency: str
    channel: str
    is_fraud: int = 0
    attack_family: str = "legitimate"


class PaymentEcosystem:
    def __init__(self, num_customers=200, num_merchants=60, num_days=90, seed=SEED):
        self.rng = random.Random(seed)
        self.num_days = num_days
        self.customers = []
        self.merchants = []
        self.accounts = []
        self.devices = []
        self.transactions = []
        self._build_merchants(num_merchants)
        self._build_customers(num_customers)

    def _build_merchants(self, n):
        for i in range(n):
            category = self.rng.choice(MERCHANT_CATEGORIES)
            city = self.rng.choice(CITIES)
            self.merchants.append(Merchant(
                merchant_id=new_id("mch"),
                name=f"{category.title().replace('_', ' ')} Store {i}",
                category=category,
                city=city,
            ))

    def _build_customers(self, n):
        start = datetime(2026, 1, 1)
        for _ in range(n):
            home_city = self.rng.choice(CITIES)
            join_offset = self.rng.randint(0, 365)
            customer = Customer(
                customer_id=new_id("cust"),
                name=f"Customer {new_id('n')[:8]}",
                home_city=home_city,
                joined_at=(start - timedelta(days=join_offset)).isoformat(),
                favorite_categories=self.rng.sample(MERCHANT_CATEGORIES, k=self.rng.randint(2, 4)),
                avg_amount=round(self.rng.uniform(300, 4000), 2),
                amount_std=round(self.rng.uniform(50, 500), 2),
                daily_txn_rate=round(self.rng.uniform(0.2, 2.0), 2),
                active_hours=self._sample_active_hours(),
                location_weights=self._sample_location_weights(home_city),
            )
            self._attach_accounts_and_devices(customer)
            self.customers.append(customer)

    def _sample_active_hours(self):
        # most customers transact in one or two windows during the day
        start_hour = self.rng.choice([7, 9, 12, 18, 20])
        return (start_hour, min(start_hour + self.rng.randint(2, 6), 23))

    def _sample_location_weights(self, home_city):
        # home city dominates, a couple of frequent cities (e.g. commute/family),
        # and a thin residual spread across the remaining cities for occasional travel
        others = [c for c in CITIES if c != home_city]
        frequent = self.rng.sample(others, k=min(2, len(others)))
        rare = [c for c in others if c not in frequent]

        home_weight = self.rng.uniform(0.70, 0.85)
        frequent_total = self.rng.uniform(0.10, 0.20)
        rare_total = max(0.0, 1.0 - home_weight - frequent_total)

        weights = {home_city: home_weight}
        for city in frequent:
            weights[city] = frequent_total / len(frequent)
        for city in rare:
            weights[city] = rare_total / len(rare) if rare else 0.0
        return weights

    def _attach_accounts_and_devices(self, customer):
        num_accounts = self.rng.choices([1, 2], weights=[0.75, 0.25])[0]
        for _ in range(num_accounts):
            account = Account(
                account_id=new_id("acc"),
                customer_id=customer.customer_id,
                account_type=self.rng.choice(ACCOUNT_TYPES),
                opened_at=customer.joined_at,
            )
            num_devices = self.rng.choices([1, 2], weights=[0.8, 0.2])[0]
            for _ in range(num_devices):
                device = Device(
                    device_id=new_id("dev"),
                    device_type=self.rng.choice(DEVICE_TYPES),
                    first_seen=customer.joined_at,
                )
                self.devices.append(device)
                account.device_ids.append(device.device_id)
            self.accounts.append(account)
            customer.account_ids.append(account.account_id)

    def _merchants_by_category(self, category):
        return [m for m in self.merchants if m.category == category]

    def _pick_merchant(self, customer, city):
        category = self.rng.choice(customer.favorite_categories)
        pool = [m for m in self._merchants_by_category(category) if m.city == city]
        if not pool:
            pool = self._merchants_by_category(category) or self.merchants
        return self.rng.choice(pool)

    def generate_transactions(self):
        start = datetime(2026, 4, 1)
        for day in range(self.num_days):
            date = start + timedelta(days=day)
            for customer in self.customers:
                count = self._poisson(customer.daily_txn_rate)
                for _ in range(count):
                    self._generate_one_transaction(customer, date)

    def _poisson(self, lam):
        # simple poisson sampler without numpy dependency
        l = 2.71828 ** (-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rng.random()
            if p <= l:
                return k - 1

    def _sample_city(self, weights):
        cities = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choices(cities, weights=probs)[0]

    def _generate_one_transaction(self, customer, date):
        city = self._sample_city(customer.location_weights)
        merchant = self._pick_merchant(customer, city)
        account_id = self.rng.choice(customer.account_ids)
        account = next(a for a in self.accounts if a.account_id == account_id)
        device_id = self.rng.choice(account.device_ids)

        hour = self.rng.randint(*customer.active_hours)
        minute = self.rng.randint(0, 59)
        timestamp = date.replace(hour=hour, minute=minute)

        amount = max(10.0, round(self.rng.gauss(customer.avg_amount, customer.amount_std), 2))
        lat, lon = self._jittered_coords(city)

        self.transactions.append(Transaction(
            transaction_id=new_id("txn"),
            customer_id=customer.customer_id,
            account_id=account_id,
            device_id=device_id,
            merchant_id=merchant.merchant_id,
            amount=amount,
            timestamp=timestamp.isoformat(),
            city=city,
            merchant_category=merchant.category,
            location_lat=lat,
            location_lon=lon,
            ip_address=self._random_ip(),
            currency=CURRENCY,
            channel=self._channel_for_device(device_id),
            is_fraud=0,
            attack_family="legitimate",
        ))

    def _device_type(self, device_id):
        device = next((d for d in self.devices if d.device_id == device_id), None)
        return device.device_type if device else self.rng.choice(DEVICE_TYPES)

    def _channel_for_device(self, device_id):
        return CHANNEL_MAP[self._device_type(device_id)]

    def _jittered_coords(self, city, spread=0.05):
        base_lat, base_lon = CITY_COORDS[city]
        return (round(base_lat + self.rng.gauss(0, spread), 5),
                round(base_lon + self.rng.gauss(0, spread), 5))

    def _random_ip(self):
        return f"10.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}"

    def to_csv(self, out_dir="."):
        self._write_csv(f"{out_dir}/customers.csv", self.customers)
        self._write_csv(f"{out_dir}/merchants.csv", self.merchants)
        self._write_csv(f"{out_dir}/accounts.csv", self.accounts)
        self._write_csv(f"{out_dir}/devices.csv", self.devices)
        self._write_csv(f"{out_dir}/transactions.csv", self.transactions)

    def _write_csv(self, path, records):
        if not records:
            return
        rows = [asdict(r) for r in records]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = {k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in row.items()}
                writer.writerow(row)


if __name__ == "__main__":
    eco = PaymentEcosystem(num_customers=200, num_merchants=60, num_days=90)
    eco.generate_transactions()
    eco.to_csv(out_dir=".")
    print(f"customers: {len(eco.customers)}")
    print(f"merchants: {len(eco.merchants)}")
    print(f"accounts: {len(eco.accounts)}")
    print(f"devices: {len(eco.devices)}")
    print(f"transactions: {len(eco.transactions)}")
