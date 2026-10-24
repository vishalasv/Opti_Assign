# 🏬 Dark Store Location & Capacity Optimization (Chennai)

A production-ready decision-support system built with **Streamlit**, **Google OR-Tools (SCIP)**, **Folium**, and **Plotly** to optimize dark store site selection, capacity utilization, and order allocation across the **Chennai Metropolitan Area**.

---

## 📌 Problem Overview & Enhancements

This project solves the **Capacitated Facility Location & Order Allocation Problem (CFLP)** formulated as a **Mixed-Integer Linear Program (MILP)**.

### Enhanced Features:
1. **Maximum Delivery Distance Constraint**: Limits maximum distance between any dark store and customer ($d_{ij} \le D_{\max}$). Orders cannot be fulfilled beyond this service radius.
2. **Chennai Geography Simulation**: Candidate dark store locations seeded across major Chennai hubs (*T. Nagar, Velachery, Anna Nagar, Adyar, Guindy, OMR, Porur, Ambattur, Tambaram, etc.*) with clustered customer demand.
3. **Interactive Visual Dashboard**:
   - Executive KPI cards and summary tables.
   - Interactive Folium map with coverage circles, opened/closed markers, customer nodes, and delivery routes.
4. **Profit Frontier Analysis (10,000 Customers & 50 Dark Stores)**:
   - Evaluates profit when opening up to at most **2, 5, 10, 20, 25** dark stores.
   - Dual-axis Plotly charts for Profit vs. Demand Fulfillment.
   - Financial breakdown of Revenue, Fixed Costs, and Transportation Costs.

---

## 📐 Mathematical Formulation

### Sets & Indices
- $I = \{1, \dots, N\}$: Set of customer locations, indexed by $i$.
- $J = \{1, \dots, M\}$: Set of candidate dark store locations, indexed by $j$.

### Parameters
- $d_{ij}$: Haversine distance in kilometers from dark store $j$ to customer $i$.
- $D_i$: Daily order demand from customer location $i$.
- $F_j$: Fixed daily cost of opening candidate dark store $j$.
- $C_j$: Daily capacity (max orders/day) of dark store $j$.
- $D_{\max}$: Maximum allowable delivery radius (km).
- $c_{ij} = \text{cost\_per\_km} \times d_{ij}$: Unit transport cost per order.
- $R$: Revenue earned per delivered order.
- $K$: Maximum number of dark stores permitted to open.

### Decision Variables
- $y_j \in \{0, 1\}$: Binary decision to open candidate dark store $j$.
- $X_{ij} \ge 0$: Continuous volume of orders from customer $i$ fulfilled by dark store $j$.

### Optimization Model
$$\max_{y, X} \quad \sum_{(i,j) \in E} (R - c_{ij}) X_{ij} - \sum_{j \in J} F_j y_j$$

**Subject to:**
1. **Max Distance Constraint**: $X_{ij} = 0 \quad \forall (i, j) \notin E \text{ where } E = \{(i, j) \mid d_{ij} \le D_{\max}\}$
2. **Customer Demand**: $\sum_{j: (i,j) \in E} X_{ij} \le D_i \quad \forall i \in I$
3. **Store Capacity**: $\sum_{i: (i,j) \in E} X_{ij} \le C_j y_j \quad \forall j \in J$
4. **Store Opening Cap**: $\sum_{j \in J} y_j \le K$
5. **Integrity & Non-negativity**: $y_j \in \{0, 1\}, \quad X_{ij} \ge 0$

---





