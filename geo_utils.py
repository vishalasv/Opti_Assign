"""
geo_utils.py: Geographic data generation and utilities for Chennai City.
Provides realistic coordinates for Candidate Dark Stores and Customers within Chennai,
along with vectorized Haversine distance computations and spatial micro-clustering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# Major commercial and residential hubs across Chennai Metropolitan Area
CHENNAI_HUBS = [
    {"name": "T. Nagar", "lat": 13.0418, "lon": 80.2341, "weight": 1.5},
    {"name": "Anna Nagar", "lat": 13.0850, "lon": 80.2101, "weight": 1.4},
    {"name": "Adyar", "lat": 13.0012, "lon": 80.2565, "weight": 1.3},
    {"name": "Velachery", "lat": 12.9815, "lon": 80.2180, "weight": 1.4},
    {"name": "Guindy", "lat": 13.0067, "lon": 80.2021, "weight": 1.2},
    {"name": "Mylapore", "lat": 13.0368, "lon": 80.2676, "weight": 1.2},
    {"name": "OMR - Sholinganallur", "lat": 12.9010, "lon": 80.2279, "weight": 1.6},
    {"name": "Porur", "lat": 13.0382, "lon": 80.1565, "weight": 1.3},
    {"name": "Tambaram", "lat": 12.9249, "lon": 80.1000, "weight": 1.3},
    {"name": "Ambattur", "lat": 13.1143, "lon": 80.1548, "weight": 1.2},
    {"name": "Kilpauk", "lat": 13.0827, "lon": 80.2417, "weight": 1.1},
    {"name": "Perambur", "lat": 13.1075, "lon": 80.2435, "weight": 1.1},
    {"name": "Koyambedu", "lat": 13.0694, "lon": 80.1948, "weight": 1.2},
    {"name": "Nungambakkam", "lat": 13.0569, "lon": 80.2425, "weight": 1.3},
    {"name": "Alwarpet", "lat": 13.0334, "lon": 80.2520, "weight": 1.1},
    {"name": "Thiruvanmiyur", "lat": 12.9830, "lon": 80.2594, "weight": 1.2},
    {"name": "Royapettah", "lat": 13.0524, "lon": 80.2612, "weight": 1.1},
    {"name": "Kodambakkam", "lat": 13.0519, "lon": 80.2212, "weight": 1.2},
    {"name": "Ashok Nagar", "lat": 13.0354, "lon": 80.2117, "weight": 1.2},
    {"name": "Saidapet", "lat": 13.0213, "lon": 80.2231, "weight": 1.1},
    {"name": "Pallavaram", "lat": 12.9675, "lon": 80.1491, "weight": 1.2},
    {"name": "Chromepet", "lat": 12.9516, "lon": 80.1462, "weight": 1.2},
    {"name": "Medavakkam", "lat": 12.9185, "lon": 80.1916, "weight": 1.3},
    {"name": "Perungudi", "lat": 12.9654, "lon": 80.2461, "weight": 1.3},
    {"name": "Madipakkam", "lat": 12.9647, "lon": 80.1961, "weight": 1.2},
]


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """
    Computes Great Circle (Haversine) distance in kilometers between an array of points
    and a single reference coordinate.
    """
    R = 6371.0  # Earth's radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_distance_matrix(
    cust_lats: np.ndarray,
    cust_lons: np.ndarray,
    store_lats: np.ndarray,
    store_lons: np.ndarray,
) -> np.ndarray:
    """
    Computes pairwise Haversine distance matrix (N_customers x M_stores) in kilometers.
    """
    n_cust = len(cust_lats)
    n_stores = len(store_lats)
    dist_matrix = np.zeros((n_cust, n_stores), dtype=np.float32)

    for j in range(n_stores):
        dist_matrix[:, j] = haversine_vectorized(cust_lats, cust_lons, store_lats[j], store_lons[j])

    return dist_matrix


def generate_chennai_data(
    n_stores: int = 4,
    n_customers: int = 10,
    random_seed: int = 42,
    base_fixed_cost: float = 3500.0,
    base_capacity: int = 400,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic but realistic dark store candidates and customer locations
    distributed across Chennai.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Candidate Dark Stores
    # Seed candidate dark stores based on Chennai hubs + realistic jitter
    store_rows = []
    hub_indices = rng.choice(len(CHENNAI_HUBS), size=n_stores, replace=(n_stores > len(CHENNAI_HUBS)))

    for idx, hub_idx in enumerate(hub_indices):
        hub = CHENNAI_HUBS[hub_idx]
        # Jitter within ~1.5 km
        lat = hub["lat"] + rng.normal(0, 0.012)
        lon = hub["lon"] + rng.normal(0, 0.012)

        # Capacity between 80% and 120% of base_capacity
        cap = int(base_capacity * rng.uniform(0.85, 1.25))
        # Fixed cost with modest variation
        f_cost = float(round(base_fixed_cost * rng.uniform(0.85, 1.20), -1))

        store_name = f"DS_{idx+1}_{hub['name'].split()[0]}"
        store_rows.append({
            "store_id": idx,
            "name": store_name,
            "hub": hub["name"],
            "lat": lat,
            "lon": lon,
            "fixed_cost": f_cost,
            "capacity": cap,
        })

    df_stores = pd.DataFrame(store_rows)

    # 2. Customers
    # Cluster customers around hubs according to hub population density weights
    weights = np.array([h["weight"] for h in CHENNAI_HUBS])
    weights = weights / weights.sum()

    chosen_hubs = rng.choice(len(CHENNAI_HUBS), size=n_customers, p=weights)
    cust_rows = []

    for i, h_idx in enumerate(chosen_hubs):
        hub = CHENNAI_HUBS[h_idx]
        # Cluster radius ~2.5 - 4 km around hub center
        lat = hub["lat"] + rng.normal(0, 0.022)
        lon = hub["lon"] + rng.normal(0, 0.022)

        # Restrict slightly to avoid ocean (Bay of Bengal is East of lon ~80.30)
        lon = min(lon, 80.295)

        # Demand per customer (1 to 5 orders)
        demand = int(rng.integers(1, 5))

        cust_rows.append({
            "customer_id": i,
            "lat": lat,
            "lon": lon,
            "demand": demand,
            "hub_area": hub["name"],
        })

    df_customers = pd.DataFrame(cust_rows)
    return df_stores, df_customers


def aggregate_customers_for_scale(
    df_customers: pd.DataFrame,
    grid_size_deg: float = 0.012,
) -> pd.DataFrame:
    """
    Aggregates large customer datasets (e.g. 10,000 customers) into spatial micro-clusters
    (grid cells of ~1.2 km) for ultra-fast solving while preserving spatial fidelity.
    """
    df = df_customers.copy()
    df["lat_bin"] = (df["lat"] / grid_size_deg).round() * grid_size_deg
    df["lon_bin"] = (df["lon"] / grid_size_deg).round() * grid_size_deg

    agg = df.groupby(["lat_bin", "lon_bin"]).agg(
        total_demand=("demand", "sum"),
        centroid_lat=("lat", "mean"),
        centroid_lon=("lon", "mean"),
        customer_count=("customer_id", "count"),
    ).reset_index()

    agg["customer_id"] = agg.index
    agg["lat"] = agg["centroid_lat"]
    agg["lon"] = agg["centroid_lon"]
    agg["demand"] = agg["total_demand"]

    return agg
