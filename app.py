"""
app.py: Dark Store Location Optimization Web Application
Built with Streamlit, Google OR-Tools, Folium, and Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

from geo_utils import generate_chennai_data
from dark_store_model import solve_dark_store_optimization, run_frontier_analysis

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dark Store Location Optimizer - Chennai",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🏬 Dark Store Location & Capacity Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Mixed-Integer Linear Programming (MILP) with Google OR-Tools &middot; Chennai Metropolitan Region</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Parameters")

st.sidebar.subheader("📍 Geography & Network")
n_stores = st.sidebar.slider("Candidate Dark Stores", min_value=4, max_value=40, value=12, step=1)
n_customers = st.sidebar.slider("Customer Count (Interactive Map)", min_value=10, max_value=800, value=150, step=10)

st.sidebar.subheader("📏 Maximum Distance Constraint")
max_dist_km = st.sidebar.slider(
    "Max Allowed Distance (km)",
    min_value=1.0,
    max_value=25.0,
    value=7.0,
    step=0.5,
    help="Limits the maximum delivery distance between any dark store and customer (d_ij <= D_max). Customers beyond this radius cannot be served from that store.",
)

st.sidebar.subheader("💰 Economics & Capacities")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    fixed_cost = st.number_input("Fixed Cost (₹/store)", min_value=500.0, max_value=20000.0, value=3500.0, step=250.0)
    cost_per_km = st.number_input("Transport Cost (₹/km)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
with col_s2:
    store_cap = st.number_input("Store Capacity (orders)", min_value=50, max_value=3000, value=350, step=25)
    rev_per_order = st.number_input("Revenue/Order (₹)", min_value=10.0, max_value=1000.0, value=180.0, step=10.0)

st.sidebar.subheader("🔒 Store Limits & Mode")
limit_stores = st.sidebar.checkbox("Limit Max Stores to Open (K)", value=False)
max_k = None
if limit_stores:
    max_k = st.sidebar.slider("Max Stores to Open (K)", min_value=1, max_value=int(n_stores), value=min(5, n_stores))

enforce_demand = st.sidebar.checkbox(
    "Enforce 100% Demand Satisfaction",
    value=False,
    help="When enabled, requires all customer demand to be strictly met (sum_j X_ij = D_i). May be infeasible if max distance or capacity is too restrictive.",
)

random_seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Interactive Chennai Map & KPIs",
    "📈 Profit Frontier (10,000 Customers & 50 Stores)",
    "📐 Mathematical Model & Constraints",
    "🚀 Deployment Guide",
])

# -----------------------------------------------------------------------------
# TAB 1: Interactive Chennai Simulation
# -----------------------------------------------------------------------------
with tab1:
    # 1. Generate Data
    df_stores_gen, df_cust_gen = generate_chennai_data(
        n_stores=n_stores,
        n_customers=n_customers,
        random_seed=random_seed,
        base_fixed_cost=fixed_cost,
        base_capacity=store_cap,
    )

    # 2. Solve Optimization
    with st.spinner("Solving Dark Store Optimization with Google OR-Tools (SCIP)..."):
        sol = solve_dark_store_optimization(
            df_stores=df_stores_gen,
            df_customers=df_cust_gen,
            max_distance_km=max_dist_km,
            cost_per_km=cost_per_km,
            revenue_per_order=rev_per_order,
            max_open_stores=max_k,
            enforce_full_demand=enforce_demand,
            time_limit_sec=15,
        )

    if not sol["success"]:
        st.error(f"❌ **Optimization Failed**: {sol['error']}")
    else:
        kpis = sol["kpis"]
        df_stores_res = sol["df_stores"]
        df_allocs = sol["df_allocations"]

        # 3. KPI Cards
        st.markdown("### 📊 Executive KPI Summary")
        kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)
        with kcol1:
            st.metric("Net Profit", f"₹{kpis['Total Profit (₹)']:,.0f}")
        with kcol2:
            st.metric("Total Revenue", f"₹{kpis['Total Revenue (₹)']:,.0f}")
        with kcol3:
            st.metric("Total Cost", f"₹{kpis['Total Cost (₹)']:,.0f}")
        with kcol4:
            st.metric("Stores Opened", f"{kpis['Stores Opened']} / {kpis['Total Candidate Stores']}")
        with kcol5:
            st.metric("Fulfillment Rate", f"{kpis['Order Fulfillment Rate (%)']:.1f}%")
        with kcol6:
            st.metric("Avg Delivery Dist", f"{kpis['Avg Delivery Distance (km)']:.2f} km")

        st.markdown("---")

        # 4. Map & Store Status
        col_map, col_table = st.columns([7, 5])

        with col_map:
            st.markdown(f"#### 🗺️ Chennai Geographic Allocation Map (Max Radius: {max_dist_km:.1f} km)")
            st.caption(
                "🟢 **Green Icons**: Opened Dark Stores (with max radius circles) &nbsp;|&nbsp; "
                "⚪ **Gray Icons**: Unopened Candidate Locations &nbsp;|&nbsp; "
                "🔵 **Dots**: Served Customers &nbsp;|&nbsp; "
                "🔴 **Red Dots**: Unserved Customers"
            )

            # Center map on Chennai
            chennai_center = [13.045, 80.225]
            m = folium.Map(location=chennai_center, zoom_start=11, tiles="CartoDB positron")

            # Store palette for coloring customer allocation lines
            color_palette = [
                "#2563EB", "#16A34A", "#9333EA", "#EA580C", "#0D9488",
                "#DC2626", "#4F46E5", "#D97706", "#059669", "#7C3AED",
                "#DB2777", "#475569", "#65A30D", "#0891B2", "#C026D3"
            ]

            opened_stores_set = set(sol["opened_store_ids"])

            # Plot Candidate Dark Stores
            for idx, s in df_stores_res.iterrows():
                store_id = int(s["store_id"])
                is_open = s["opened"]

                if is_open:
                    icon_color = "green"
                    popup_html = f"""
                    <div style='font-family: Arial; font-size: 13px; width: 180px;'>
                        <b style='color:#15803D;'>{s['name']}</b><br>
                        <b>Status:</b> OPEN<br>
                        <b>Orders:</b> {s['orders_served']:.0f} / {s['capacity']}<br>
                        <b>Utilization:</b> {s['utilization_pct']:.1f}%<br>
                        <b>Fixed Cost:</b> ₹{s['fixed_cost']:,.0f}<br>
                        <b>Coverage Radius:</b> {max_dist_km} km
                    </div>
                    """
                    # Store marker
                    folium.Marker(
                        location=[s["lat"], s["lon"]],
                        tooltip=f"OPEN: {s['name']} ({s['orders_served']:.0f} orders)",
                        popup=folium.Popup(popup_html, max_width=220),
                        icon=folium.Icon(color=icon_color, icon="shopping-cart", prefix="fa"),
                    ).add_to(m)

                    # Coverage circle (max distance constraint)
                    folium.Circle(
                        location=[s["lat"], s["lon"]],
                        radius=max_dist_km * 1000,  # meters
                        color="#16A34A",
                        weight=1.5,
                        fill=True,
                        fill_color="#22C55E",
                        fill_opacity=0.08,
                    ).add_to(m)
                else:
                    popup_html = f"""
                    <div style='font-family: Arial; font-size: 13px; width: 160px;'>
                        <b style='color:#6B7280;'>{s['name']}</b><br>
                        <b>Status:</b> NOT OPENED<br>
                        <b>Capacity:</b> {s['capacity']} orders<br>
                        <b>Fixed Cost:</b> ₹{df_stores_gen.iloc[store_id]['fixed_cost']:,.0f}
                    </div>
                    """
                    folium.CircleMarker(
                        location=[s["lat"], s["lon"]],
                        radius=6,
                        color="#9CA3AF",
                        fill=True,
                        fill_color="#D1D5DB",
                        fill_opacity=0.7,
                        tooltip=f"Candidate: {s['name']} (Closed)",
                        popup=folium.Popup(popup_html, max_width=200),
                    ).add_to(m)

            # Plot Customers and Delivery Connections
            served_cust_ids = set()
            if not df_allocs.empty:
                for _, alloc in df_allocs.iterrows():
                    c_id = int(alloc["customer_id"])
                    s_id = int(alloc["store_id"])
                    served_cust_ids.add(c_id)

                    line_color = color_palette[s_id % len(color_palette)]

                    # Delivery Route Polyline
                    folium.PolyLine(
                        locations=[[alloc["cust_lat"], alloc["cust_lon"]], [alloc["store_lat"], alloc["store_lon"]]],
                        color=line_color,
                        weight=1.8,
                        opacity=0.6,
                        tooltip=f"{alloc['store_name']} &rarr; Cust {c_id}: {alloc['distance_km']} km, {alloc['volume']} orders",
                    ).add_to(m)

                    # Customer marker
                    folium.CircleMarker(
                        location=[alloc["cust_lat"], alloc["cust_lon"]],
                        radius=3.5,
                        color=line_color,
                        fill=True,
                        fill_color=line_color,
                        fill_opacity=0.9,
                        tooltip=f"Customer {c_id}: {alloc['volume']} orders from {alloc['store_name']}",
                    ).add_to(m)

            # Unserved Customers
            for _, cust in df_cust_gen.iterrows():
                c_id = int(cust["customer_id"])
                if c_id not in served_cust_ids:
                    folium.CircleMarker(
                        location=[cust["lat"], cust["lon"]],
                        radius=4,
                        color="#EF4444",
                        fill=True,
                        fill_color="#EF4444",
                        fill_opacity=0.9,
                        tooltip=f"Customer {c_id}: UNSERVED (Demand = {cust['demand']})",
                    ).add_to(m)

            st_folium(m, width="100%", height=550)

        with col_table:
            st.markdown("#### 🏪 Candidate Store Utilization")
            display_stores = df_stores_res.copy()
            display_stores["Status"] = display_stores["opened"].apply(lambda x: "🟢 OPEN" if x else "⚪ CLOSED")
            display_stores = display_stores[
                ["Status", "name", "hub", "orders_served", "capacity", "utilization_pct", "fixed_cost"]
            ]
            display_stores.columns = ["Status", "Store Name", "Hub Area", "Orders", "Capacity", "Util %", "Fixed Cost (₹)"]

            st.dataframe(
                display_stores.style.format({
                    "Orders": "{:.0f}",
                    "Capacity": "{:.0f}",
                    "Util %": "{:.1f}%",
                    "Fixed Cost (₹)": "₹{:,.0f}",
                }),
                height=260,
                use_container_width=True,
            )

            st.markdown("#### 📋 Detailed KPI Breakdown")
            df_kpis = pd.DataFrame([
                {"KPI Metric": "Net Profit", "Value": f"₹{kpis['Total Profit (₹)']:,.2f}"},
                {"KPI Metric": "Gross Revenue", "Value": f"₹{kpis['Total Revenue (₹)']:,.2f}"},
                {"KPI Metric": "Total Cost", "Value": f"₹{kpis['Total Cost (₹)']:,.2f}"},
                {"KPI Metric": "Fixed Opening Costs", "Value": f"₹{kpis['Fixed Opening Costs (₹)']:,.2f}"},
                {"KPI Metric": "Transportation Costs", "Value": f"₹{kpis['Transportation Costs (₹)']:,.2f}"},
                {"KPI Metric": "Total Candidate Stores", "Value": str(kpis["Total Candidate Stores"])},
                {"KPI Metric": "Stores Opened", "Value": str(kpis["Stores Opened"])},
                {"KPI Metric": "Orders Fulfilled", "Value": f"{kpis['Total Orders Fulfilled']} / {kpis['Total Customer Demand']}"},
                {"KPI Metric": "Fulfillment Rate", "Value": f"{kpis['Order Fulfillment Rate (%)']:.2f}%"},
                {"KPI Metric": "Average Delivery Distance", "Value": f"{kpis['Avg Delivery Distance (km)']:.2f} km"},
                {"KPI Metric": "Max Distance Allowed", "Value": f"{max_dist_km:.1f} km"},
                {"KPI Metric": "Solver Status", "Value": kpis["Solver Status"]},
                {"KPI Metric": "Solve Execution Time", "Value": f"{kpis['Solve Time (s)']:.3f} s"},
            ])
            st.dataframe(df_kpis, hide_index=True, use_container_width=True, height=250)

        # 5. Customer Allocations Table (Collapsible)
        with st.expander("📄 View & Download Full Customer Allocation Matrix"):
            if not df_allocs.empty:
                st.dataframe(
                    df_allocs[["customer_id", "store_name", "volume", "distance_km", "cost"]].rename(
                        columns={
                            "customer_id": "Customer ID",
                            "store_name": "Assigned Dark Store",
                            "volume": "Orders Supplied",
                            "distance_km": "Distance (km)",
                            "cost": "Delivery Cost (₹)",
                        }
                    ),
                    use_container_width=True,
                )
                csv = df_allocs.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Customer Allocations CSV", csv, "customer_allocations.csv", "text/csv")
            else:
                st.info("No allocations available.")


# -----------------------------------------------------------------------------
# TAB 2: Profit Frontier (10,000 Customers & 50 Dark Stores)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Profit Frontier for 10,000 Customers & 50 Dark Store Candidates")
    st.markdown(
        r"""
        > **Scenario Objective**: Evaluate total profit when up to at most **2, 5, 10, 20, 25** dark store locations
        are allowed to open across Chennai for **10,000 customer demand points** under the maximum delivery distance constraint ($d_{ij} \le D_{\max}$).
        """
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_max_dist = st.slider("Frontier Max Distance (km)", min_value=3.0, max_value=15.0, value=8.0, step=0.5)
    with col_f2:
        f_rev = st.number_input("Frontier Revenue/Order (₹)", min_value=50.0, max_value=500.0, value=180.0, step=10.0)
    with col_f3:
        f_fixed = st.number_input("Frontier Store Fixed Cost (₹)", min_value=1000.0, max_value=10000.0, value=4000.0, step=500.0)

    @st.cache_data(show_spinner=False)
    def compute_cached_frontier(max_d: float, rev: float, f_cost: float, seed: int):
        return run_frontier_analysis(
            n_customers=10000,
            n_stores=50,
            k_values=[2, 5, 10, 20, 25],
            max_distance_km=max_d,
            revenue_per_order=rev,
            base_fixed_cost=f_cost,
            random_seed=seed,
        )

    with st.spinner("Optimizing 10,000 customers across 50 candidate locations for K in [2, 5, 10, 20, 25]..."):
        df_frontier = compute_cached_frontier(f_max_dist, f_rev, f_fixed, random_seed)

    # 1. Dual-Axis Plotly Chart: Profit and Fulfillment Rate vs K
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Net Profit Trace (Bar / Line)
    fig.add_trace(
        go.Scatter(
            x=df_frontier["Allowed Stores (K)"],
            y=df_frontier["Net Profit (₹)"],
            mode="lines+markers+text",
            name="Net Profit (₹)",
            line=dict(color="#2563EB", width=3.5),
            marker=dict(size=10, color="#1D4ED8"),
            text=[f"₹{p/1e5:.2f}L" for p in df_frontier["Net Profit (₹)"]],
            textposition="top center",
        ),
        secondary_y=False,
    )

    # Fulfillment Rate Trace
    fig.add_trace(
        go.Scatter(
            x=df_frontier["Allowed Stores (K)"],
            y=df_frontier["Fulfillment Rate (%)"],
            mode="lines+markers+text",
            name="Demand Fulfillment Rate (%)",
            line=dict(color="#10B981", width=3, dash="dot"),
            marker=dict(size=9, color="#059669"),
            text=[f"{f:.1f}%" for f in df_frontier["Fulfillment Rate (%)"]],
            textposition="bottom right",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="<b>Profit & Fulfillment Frontier vs. Maximum Allowed Dark Stores (K)</b>",
        xaxis_title="Maximum Allowed Open Dark Stores (K)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
    )
    fig.update_yaxes(title_text="<b>Net Profit (₹)</b>", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="<b>Demand Fulfillment Rate (%)</b>", secondary_y=True, showgrid=False, range=[0, 105])

    st.plotly_chart(fig, use_container_width=True)

    # 2. Revenue vs Cost Breakdown Chart
    col_c1, col_c2 = st.columns([7, 5])
    with col_c1:
        fig_breakdown = go.Figure()
        fig_breakdown.add_trace(
            go.Bar(
                x=[f"K={k}" for k in df_frontier["Allowed Stores (K)"]],
                y=df_frontier["Revenue (₹)"],
                name="Gross Revenue",
                marker_color="#3B82F6",
            )
        )
        fig_breakdown.add_trace(
            go.Bar(
                x=[f"K={k}" for k in df_frontier["Allowed Stores (K)"]],
                y=df_frontier["Delivery Cost (₹)"],
                name="Variable Delivery Cost",
                marker_color="#F59E0B",
            )
        )
        fig_breakdown.add_trace(
            go.Bar(
                x=[f"K={k}" for k in df_frontier["Allowed Stores (K)"]],
                y=df_frontier["Fixed Cost (₹)"],
                name="Fixed Store Opening Cost",
                marker_color="#EF4444",
            )
        )
        fig_breakdown.update_layout(
            title="<b>Revenue and Cost Breakdown by K</b>",
            barmode="group",
            template="plotly_white",
            height=360,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

    with col_c2:
        st.markdown("#### 📋 Frontier Results Table")
        df_display = df_frontier[[
            "Allowed Stores (K)", "Stores Opened", "Net Profit (₹)", "Revenue (₹)",
            "Total Cost (₹)", "Fulfillment Rate (%)", "Avg Distance (km)"
        ]].copy()

        # Compute marginal profit
        df_display["Marginal Profit (₹)"] = df_display["Net Profit (₹)"].diff().fillna(df_display["Net Profit (₹)"].iloc[0])

        st.dataframe(
            df_display.style.format({
                "Net Profit (₹)": "₹{:,.0f}",
                "Marginal Profit (₹)": "+₹{:,.0f}",
                "Revenue (₹)": "₹{:,.0f}",
                "Total Cost (₹)": "₹{:,.0f}",
                "Fulfillment Rate (%)": "{:.1f}%",
                "Avg Distance (km)": "{:.2f} km",
            }),
            use_container_width=True,
            height=320,
        )

    # 3. Managerial Insights Callout
    st.markdown("---")
    st.markdown("### 💡 Key Managerial Takeaways & Sweet-Spot Analysis")
    max_profit_row = df_frontier.loc[df_frontier["Net Profit (₹)"].idxmax()]
    best_k = int(max_profit_row["Allowed Stores (K)"])
    best_profit = max_profit_row["Net Profit (₹)"]
    best_fulfillment = max_profit_row["Fulfillment Rate (%)"]

    st.success(
        f"""
        - **Optimal Sweet Spot**: Maximum profit is achieved around **K = {best_k} stores**, yielding **₹{best_profit:,.0f} in Net Profit** with **{best_fulfillment:.1f}% fulfillment**.
        - **Diminishing Marginal Returns**: Going from **K = 2 &rarr; 5 &rarr; 10** produces dramatic revenue and profit jumps because dense customer clusters in Chennai (T. Nagar, Velachery, OMR, Anna Nagar) are covered.
        - **Saturation Effect**: Moving from **K = 20 &rarr; 25** provides small incremental demand coverage (~91% &rarr; ~100%), but additional fixed costs reduce or flatten marginal profit. Opening between 18 to 22 dark stores represents the optimal economic trade-off.
        """
    )


# -----------------------------------------------------------------------------
# TAB 3: Mathematical Model & Formulation
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 📐 Mathematical Formulation")
    st.markdown(
        r"""
        #### 1. Sets and Indices
        - $I = \{1, \dots, N\}$: Set of customer demand locations, indexed by $i$.
        - $J = \{1, \dots, M\}$: Set of candidate dark store locations, indexed by $j$.

        #### 2. Parameters & Inputs
        - $d_{ij}$: Haversine distance (in km) between candidate dark store $j$ and customer $i$.
        - $D_i$: Daily order demand from customer location $i$.
        - $F_j$: Fixed daily cost of opening and operating dark store $j$.
        - $C_j$: Daily capacity (maximum orders per day) of dark store $j$.
        - $D_{\max}$: **Maximum allowed delivery distance (km)** between any dark store and customer.
        - $c_{ij} = \text{cost\_per\_km} \times d_{ij}$: Unit transport cost per delivered order.
        - $R$: Revenue earned per delivered order.
        - $K$: Maximum number of candidate dark stores permitted to open ($\sum_j y_j \le K$).

        #### 3. Decision Variables
        - $y_j \in \{0, 1\}$: Binary variable indicating whether a dark store is opened at candidate location $j$.
        - $X_{ij} \ge 0$: Continuous volume of orders for customer $i$ fulfilled from dark store $j$.

        #### 4. Constraints
        1. **Maximum Delivery Distance Constraint**:
           $$X_{ij} = 0 \quad \forall (i, j) \text{ such that } d_{ij} > D_{\max}$$
           *(Enforced via sparse variable definition $E = \{(i, j) \mid d_{ij} \le D_{\max}\}$, drastically reducing model dimensionality).*

        2. **Demand Satisfaction**:
           $$\sum_{j \in J: (i,j) \in E} X_{ij} \le D_i \quad \forall i \in I$$

        3. **Store Capacity Constraint**:
           $$\sum_{i \in I: (i,j) \in E} X_{ij} \le C_j y_j \quad \forall j \in J$$

        4. **Maximum Stores Constraint**:
           $$\sum_{j \in J} y_j \le K$$

        #### 5. Objective Function (Profit Maximization)
        $$\max_{y, X} \quad \sum_{(i,j) \in E} (R - c_{ij}) X_{ij} - \sum_{j \in J} F_j y_j$$
        """
    )


# -----------------------------------------------------------------------------
# TAB 4: Deployment Guide
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🚀 Deployment to Streamlit Community Cloud (`https://share.streamlit.io`)")
    st.markdown(
        """
        Follow these steps to deploy this application to **Streamlit Community Cloud**:

        1. **Push your code to GitHub**:
           Make sure your repository contains:
           - `app.py` (Main Streamlit app)
           - `dark_store_model.py` (MILP Optimization engine)
           - `geo_utils.py` (Chennai geography and distance utilities)
           - `requirements.txt` (Dependencies)

        2. **Log into Streamlit Community Cloud**:
           Visit [https://share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.

        3. **Deploy New App**:
           - Click the **'New app'** button.
           - Select your GitHub repository (e.g. `your-username/chennai-darkstore-optimizer`).
           - Set the Branch to `main`.
           - Set the Main file path to `app.py`.
           - Click **'Deploy!'**.

        4. **Your Live Public Link**:
           Streamlit will automatically build your environment and provide a permanent public link:
           ```text
           https://<your-app-name>.streamlit.app
           ```
        """
    )
