"""
dark_store_model.py: Core optimization engine for Dark Store Location Optimization using Google OR-Tools.
Implements the Mixed-Integer Linear Program (MILP) with:
- Maximum delivery distance constraint (d_ij <= D_max)
- Store capacity limits (sum_i X_ij <= C_j * y_j)
- Customer demand satisfaction / profit maximization
- Parametric limit on opened stores (sum_j y_j <= K)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from ortools.linear_solver import pywraplp

from geo_utils import (
    compute_distance_matrix,
    generate_chennai_data,
    aggregate_customers_for_scale,
)


def solve_dark_store_optimization(
    df_stores: pd.DataFrame,
    df_customers: pd.DataFrame,
    max_distance_km: float = 7.0,
    cost_per_km: float = 5.0,
    revenue_per_order: float = 180.0,
    max_open_stores: Optional[int] = None,
    enforce_full_demand: bool = False,
    time_limit_sec: int = 15,
) -> Dict[str, Any]:
    """
    Formulates and solves the Dark Store Location Optimization problem.
    """
    n_stores = len(df_stores)
    n_customers = len(df_customers)

    # Compute pairwise distance matrix
    dist_matrix = compute_distance_matrix(
        cust_lats=df_customers["lat"].values,
        cust_lons=df_customers["lon"].values,
        store_lats=df_stores["lat"].values,
        store_lons=df_stores["lon"].values,
    )

    # Initialize OR-Tools SCIP solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return {"success": False, "error": "SCIP solver backend is unavailable."}

    if time_limit_sec > 0:
        solver.SetTimeLimit(int(time_limit_sec * 1000))

    # --- 1. Decision Variables ---
    # y_j: Binary variable (1 if candidate store j is opened, 0 otherwise)
    y = [solver.BoolVar(f"y_{j}") for j in range(n_stores)]

    # X_ij: Continuous variable representing orders fulfilled by store j for customer i
    # Maximum Distance Constraint:
    # Only instantiate X_ij if dist_matrix[i, j] <= max_distance_km
    X: Dict[tuple, Any] = {}
    valid_pairs_per_customer: Dict[int, List[int]] = {i: [] for i in range(n_customers)}
    valid_customers_per_store: Dict[int, List[int]] = {j: [] for j in range(n_stores)}

    demands = df_customers["demand"].values.astype(float)
    capacities = df_stores["capacity"].values.astype(float)
    fixed_costs = df_stores["fixed_cost"].values.astype(float)

    for i in range(n_customers):
        d_i = demands[i]
        for j in range(n_stores):
            d_ij = float(dist_matrix[i, j])
            if d_ij <= max_distance_km:
                var = solver.NumVar(0.0, d_i, f"X_{i}_{j}")
                X[(i, j)] = var
                valid_pairs_per_customer[i].append(j)
                valid_customers_per_store[j].append(i)

    # --- 2. Constraints ---
    # Constraint A: Demand satisfaction
    for i in range(n_customers):
        stores_for_i = valid_pairs_per_customer[i]
        if len(stores_for_i) > 0:
            expr = sum(X[(i, j)] for j in stores_for_i)
            if enforce_full_demand:
                solver.Add(expr == demands[i])
            else:
                solver.Add(expr <= demands[i])
        else:
            if enforce_full_demand:
                # Impossible to fulfill full demand because no store is within max_distance_km
                return {
                    "success": False,
                    "error": f"Customer {i} has no candidate dark store within the maximum distance of {max_distance_km:.1f} km. Relax the maximum distance constraint or disable 'Enforce 100% Demand Satisfaction'.",
                }

    # Constraint B: Capacity constraint: sum_i X_ij <= C_j * y_j
    for j in range(n_stores):
        custs_for_j = valid_customers_per_store[j]
        if len(custs_for_j) > 0:
            solver.Add(sum(X[(i, j)] for i in custs_for_j) <= capacities[j] * y[j])
        else:
            # If no customers are within reach, store shouldn't open
            solver.Add(y[j] == 0)

    # Constraint C: Maximum open stores limit (if provided)
    if max_open_stores is not None and max_open_stores > 0:
        solver.Add(sum(y[j] for j in range(n_stores)) <= int(max_open_stores))

    # --- 3. Objective Function ---
    objective = solver.Objective()

    # If enforce_full_demand, minimize total cost: sum (c_ij * X_ij) + sum (F_j * y_j)
    # Otherwise, maximize net profit: sum (Revenue - c_ij) * X_ij - sum (F_j * y_j)
    if enforce_full_demand:
        for (i, j), var in X.items():
            d_ij = float(dist_matrix[i, j])
            delivery_cost = cost_per_km * d_ij
            objective.SetCoefficient(var, delivery_cost)
        for j in range(n_stores):
            objective.SetCoefficient(y[j], fixed_costs[j])
        objective.SetMinimization()
    else:
        for (i, j), var in X.items():
            d_ij = float(dist_matrix[i, j])
            delivery_cost = cost_per_km * d_ij
            unit_margin = revenue_per_order - delivery_cost
            objective.SetCoefficient(var, unit_margin)
        for j in range(n_stores):
            objective.SetCoefficient(y[j], -fixed_costs[j])
        objective.SetMaximization()

    # --- 4. Solve ---
    t_start = time.time()
    solve_status = solver.Solve()
    solve_time = time.time() - t_start

    is_optimal = solve_status == pywraplp.Solver.OPTIMAL
    is_feasible = solve_status == pywraplp.Solver.FEASIBLE

    if not (is_optimal or is_feasible):
        return {
            "success": False,
            "error": "Optimization could not find a feasible solution under the given constraints. Try increasing store capacity or max distance.",
        }

    # --- 5. Extract Results ---
    # Store decisions
    store_results = []
    opened_store_ids = []

    for j in range(n_stores):
        is_open = bool(round(y[j].solution_value() > 0.5))
        custs_for_j = valid_customers_per_store[j]
        served_orders = sum(X[(i, j)].solution_value() for i in custs_for_j)
        cap = capacities[j]
        utilization = (served_orders / cap * 100.0) if cap > 0 else 0.0

        if is_open:
            opened_store_ids.append(j)

        store_results.append({
            "store_id": j,
            "name": df_stores.iloc[j]["name"],
            "hub": df_stores.iloc[j].get("hub", "Chennai"),
            "lat": df_stores.iloc[j]["lat"],
            "lon": df_stores.iloc[j]["lon"],
            "opened": is_open,
            "capacity": cap,
            "orders_served": round(served_orders, 1),
            "utilization_pct": round(utilization, 1),
            "fixed_cost": fixed_costs[j] if is_open else 0.0,
        })

    df_store_results = pd.DataFrame(store_results)

    # Customer allocations
    total_fulfilled_demand = 0.0
    total_delivery_cost = 0.0
    weighted_distance_sum = 0.0
    allocations = []

    for (i, j), var in X.items():
        val = var.solution_value()
        if val > 1e-4:
            d_ij = float(dist_matrix[i, j])
            d_cost = cost_per_km * d_ij * val
            total_fulfilled_demand += val
            total_delivery_cost += d_cost
            weighted_distance_sum += d_ij * val

            allocations.append({
                "customer_id": i,
                "cust_lat": df_customers.iloc[i]["lat"],
                "cust_lon": df_customers.iloc[i]["lon"],
                "store_id": j,
                "store_name": df_stores.iloc[j]["name"],
                "store_lat": df_stores.iloc[j]["lat"],
                "store_lon": df_stores.iloc[j]["lon"],
                "volume": round(val, 2),
                "distance_km": round(d_ij, 2),
                "cost": round(d_cost, 2),
            })

    df_allocations = pd.DataFrame(allocations)

    total_demand = float(demands.sum())
    total_fixed_cost = float(df_store_results["fixed_cost"].sum())
    total_revenue = float(total_fulfilled_demand * revenue_per_order)
    total_cost = float(total_fixed_cost + total_delivery_cost)
    net_profit = float(total_revenue - total_cost)
    fulfillment_rate = (total_fulfilled_demand / total_demand * 100.0) if total_demand > 0 else 0.0
    avg_delivery_dist = (weighted_distance_sum / total_fulfilled_demand) if total_fulfilled_demand > 0 else 0.0

    kpi_summary = {
        "Total Profit (₹)": net_profit,
        "Total Revenue (₹)": total_revenue,
        "Total Cost (₹)": total_cost,
        "Fixed Opening Costs (₹)": total_fixed_cost,
        "Transportation Costs (₹)": total_delivery_cost,
        "Stores Opened": len(opened_store_ids),
        "Total Candidate Stores": n_stores,
        "Total Orders Fulfilled": round(total_fulfilled_demand),
        "Total Customer Demand": round(total_demand),
        "Order Fulfillment Rate (%)": round(fulfillment_rate, 2),
        "Avg Delivery Distance (km)": round(avg_delivery_dist, 2),
        "Solver Status": "OPTIMAL" if is_optimal else "FEASIBLE",
        "Solve Time (s)": round(solve_time, 3),
    }

    return {
        "success": True,
        "kpis": kpi_summary,
        "df_stores": df_store_results,
        "df_allocations": df_allocations,
        "opened_store_ids": opened_store_ids,
        "solve_time": solve_time,
    }


def run_frontier_analysis(
    n_customers: int = 10000,
    n_stores: int = 50,
    k_values: List[int] = [2, 5, 10, 20, 25],
    max_distance_km: float = 8.0,
    cost_per_km: float = 5.0,
    revenue_per_order: float = 180.0,
    base_fixed_cost: float = 4000.0,
    base_capacity: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Solves the 10,000 customers & 50 dark store locations scenario for
    K in [2, 5, 10, 20, 25] open stores to construct the profit frontier.
    Uses spatial micro-clustering for interactive performance.
    """
    # 1. Generate full scale dataset
    df_stores, df_customers_raw = generate_chennai_data(
        n_stores=n_stores,
        n_customers=n_customers,
        random_seed=random_seed,
        base_fixed_cost=base_fixed_cost,
        base_capacity=base_capacity,
    )

    # 2. Micro-cluster customers for high-performance solving
    df_customers_clustered = aggregate_customers_for_scale(df_customers_raw, grid_size_deg=0.012)

    results = []
    for k in k_values:
        res = solve_dark_store_optimization(
            df_stores=df_stores,
            df_customers=df_customers_clustered,
            max_distance_km=max_distance_km,
            cost_per_km=cost_per_km,
            revenue_per_order=revenue_per_order,
            max_open_stores=k,
            enforce_full_demand=False,
            time_limit_sec=20,
        )

        if res["success"]:
            kpis = res["kpis"]
            results.append({
                "Allowed Stores (K)": k,
                "Stores Opened": kpis["Stores Opened"],
                "Net Profit (₹)": kpis["Total Profit (₹)"],
                "Revenue (₹)": kpis["Total Revenue (₹)"],
                "Fixed Cost (₹)": kpis["Fixed Opening Costs (₹)"],
                "Delivery Cost (₹)": kpis["Transportation Costs (₹)"],
                "Total Cost (₹)": kpis["Total Cost (₹)"],
                "Orders Fulfilled": kpis["Total Orders Fulfilled"],
                "Fulfillment Rate (%)": kpis["Order Fulfillment Rate (%)"],
                "Avg Distance (km)": kpis["Avg Delivery Distance (km)"],
                "Solve Time (s)": res["solve_time"],
            })

    return pd.DataFrame(results)
