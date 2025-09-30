# Advanced_Vehicle_Sales_App_Simple_CityView_v4.py
# ---------------------------------------------------
# 🚗 Vehicle Sales Insights — City View v4
# - Includes Market Share & Customer Segmentation Analysis
# - Top-3 models not duplicated in Least-3
# - Reasons diversified, price tiers, filters retained
# ---------------------------------------------------

import sys
import streamlit as st
import pandas as pd
import numpy as np
import io
import random
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Compatibility helpers
# ---------------------------
def get_ohe():
    try:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def compute_rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------------------
# App config & CSS
# ---------------------------
st.set_page_config(page_title="🚗 Vehicle Sales Insights (City View)", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp {
      background: linear-gradient(180deg, #0b1220 0%, #08101a 100%);
        color: #e6eef6; 
        }
    .big-title { 
    font-size:28px; 
    font-weight:700;
      }
    .muted { 
    color: #b8c6d9; 
    font-size:13px;
      }
    .card {
      background-color: 
      rgba(255,255,255,0.05); 
      padding: 14px;
        border-radius: 10px; 
        margin-bottom: 14px; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Utilities
# ---------------------------
REQUIRED_COLUMNS = {"City", "Model_name", "Category", "model_type", "fuel", "selling_price", "Units_Sold"}

@st.cache_data
def generate_sample_dataset(n=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    cities = ["mumbai","pune","delhi","bangalore","chennai","shimla","leh"]
    models = [f"Model_{i}" for i in range(1, 31)]
    categories = ["SUV","Sedan","Hatchback","MPV","Truck"]
    model_types = ["Manual","Automatic"]
    fuels = ["Petrol","Diesel","Electric","CNG"]
    rows = []
    for _ in range(n):
        city = random.choice(cities)
        model = random.choice(models)
        category = random.choice(categories)
        mtype = random.choice(model_types)
        fuel = random.choice(fuels)
        price = int(abs(np.random.normal(loc=800000, scale=220000)))
        units = int(max(0, np.random.poisson(lam=8 if price < 900000 else 3)))
        date = datetime(random.randint(2021,2024), random.randint(1,12), random.randint(1,28)).strftime("%Y-%m-%d")
        rows.append([city, model, category, mtype, fuel, price, units, date])
    df = pd.DataFrame(rows, columns=["City","Model_name","Category","model_type","fuel","selling_price","Units_Sold","Date"])
    return df

def validate_dataset(df: pd.DataFrame):
    present = set(df.columns.str.strip())
    missing = REQUIRED_COLUMNS - present
    return missing

def format_inr(value):
    try:
        v = int(round(value))
        return f"₹{v:,}"
    except Exception:
        return str(value)

def add_price_tier(df: pd.DataFrame):
    def tier(p):
        try:
            p = float(p)
            if p < 500000:
                return "Budget"
            elif p <= 1200000:
                return "Mid"
            else:
                return "Premium"
        except Exception:
            return "Unknown"
    df["Price_Tier"] = df["selling_price"].apply(tier)
    return df

REASON_TEMPLATES = [
    "Competitive pricing compared to rivals",
    "Strong resale value in the local market",
    "Better mileage and lower fuel costs",
    "High demand due to compact size and parking ease",
    "Trusted brand reputation and reliability",
    "Better service network and availability of parts",
    "Suitable for rough terrain and city roads",
    "Advanced safety and comfort features",
    "Growing adoption of eco-friendly vehicles",
    "Maintenance costs are relatively low",
    "Stylish design attracts younger buyers",
    "Limited charging infrastructure affected sales",
    "High upfront cost reduced affordability",
    "Larger size makes it less convenient in city traffic",
    "Better suited for highways than congested city roads"
]

def assign_reasons(models):
    templates = REASON_TEMPLATES.copy()
    random.shuffle(templates)
    return {m: templates[i % len(templates)] for i, m in enumerate(models)}

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Data & Options")
    uploaded_file = st.file_uploader("Upload dataset (.xlsx or .csv)", type=["xlsx","csv"])
    use_sample = st.checkbox("Use sample dataset", value=False)
    st.markdown("---")
    st.caption("Required: City, Model_name, Category, model_type, fuel, selling_price, Units_Sold")

# ---------------------------
# Load data
# ---------------------------
df = None
if use_sample and not uploaded_file:
    df = generate_sample_dataset(400)
    st.sidebar.success("Loaded sample dataset (400 rows)")
elif uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded `{uploaded_file.name}` ({df.shape[0]} rows)")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.stop()
        sys.exit(1)

if df is None:
    st.title("🚗 Vehicle Sales Insights (City View)")
    st.write("Upload a dataset (or enable **Use sample dataset**) to start.")
    st.info("Required columns: City, Model_name, Category, model_type, fuel, selling_price, Units_Sold")
    st.stop()
    sys.exit(0)

# ---------------------------
# Clean & validate
# ---------------------------
df.columns = df.columns.str.strip()
if "City" in df.columns:
    df["City"] = df["City"].astype(str).str.strip().str.lower()
if "Date" in df.columns:
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass

missing = validate_dataset(df)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()
    sys.exit(1)

for c in ["selling_price","Units_Sold"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
for c in ["City","Model_name","Category","model_type","fuel"]:
    if c in df.columns:
        df[c] = df[c].astype(str).fillna("unknown").str.strip()

df["Revenue"] = df["selling_price"] * df["Units_Sold"]
df = add_price_tier(df)

# ---------------------------
# Global Filters
# ---------------------------
st.markdown("---")
st.subheader("Global Filters (affect displayed tables/plots)")
with st.expander("Adjust filters", expanded=False):
    c1, c2, c3 = st.columns(3)
    cities = sorted(df["City"].unique())
    sel_cities = c1.multiselect("City (filter)", cities, default=cities[:5])
    cats = sorted(df["Category"].unique())
    sel_cats = c2.multiselect("Category (filter)", cats, default=cats)
    fuels = sorted(df["fuel"].unique())
    sel_fuels = c3.multiselect("Fuel (filter)", fuels, default=fuels)
    min_p, max_p = int(df["selling_price"].min()), int(df["selling_price"].max())
    price_range = st.slider("Price range", min_p, max_p, (min_p, max_p))

filtered = df[
    (df["City"].isin(sel_cities if sel_cities else df["City"].unique())) &
    (df["Category"].isin(sel_cats if sel_cats else df["Category"].unique())) &
    (df["fuel"].isin(sel_fuels if sel_fuels else df["fuel"].unique())) &
    (df["selling_price"] >= price_range[0]) &
    (df["selling_price"] <= price_range[1])
].copy()

# ---------------------------
# Tabs for sections
# ---------------------------
tabs = st.tabs(["🏙 City Summary", "📊 Market Share", "👥 Customer Segments", "🔍 Drilldown"])

# ---------------------------
# City Summary
# ---------------------------
with tabs[0]:
    st.markdown('<div class="big-title">🚗 Vehicle Sales Insights — City Summary</div>', unsafe_allow_html=True)
    city_summary = df.groupby("City", as_index=False).agg(
        Total_Units_Sold=("Units_Sold","sum"),
        Total_Revenue=("Revenue","sum"),
        Avg_Selling_Price=("selling_price","mean"),
        Top_Category=("Category", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
        Top_Fuel=("fuel", lambda x: x.mode()[0] if not x.mode().empty else "N/A")
    )
    city_summary["Total_Revenue"] = city_summary["Total_Revenue"].apply(lambda x: format_inr(x))
    city_summary["Avg_Selling_Price"] = city_summary["Avg_Selling_Price"].apply(lambda x: format_inr(x))
    st.dataframe(city_summary.sort_values("Total_Units_Sold", ascending=False).reset_index(drop=True), width="stretch")

# ---------------------------
# Market Share Analysis
# ---------------------------
with tabs[1]:
    st.markdown('<div class="big-title">📊 Market Share Analysis</div>', unsafe_allow_html=True)
    ms_city = filtered.groupby("City", as_index=False)["Units_Sold"].sum()
    ms_city["Share"] = (ms_city["Units_Sold"] / ms_city["Units_Sold"].sum() * 100).round(1)
    fig1 = px.pie(ms_city, names="City", values="Units_Sold", title="Market Share by City")
    st.plotly_chart(fig1, width="stretch")

    ms_cat = filtered.groupby("Category", as_index=False)["Units_Sold"].sum()
    fig2 = px.bar(ms_cat, x="Category", y="Units_Sold", color="Category", title="Market Share by Category")
    st.plotly_chart(fig2, width="stretch")

# ---------------------------
# Customer Segmentation
# ---------------------------
with tabs[2]:
    st.markdown('<div class="big-title">👥 Customer Segment Analysis</div>', unsafe_allow_html=True)
    seg1 = filtered.groupby("Price_Tier", as_index=False)["Units_Sold"].sum()
    fig3 = px.pie(seg1, names="Price_Tier", values="Units_Sold", title="Sales Distribution by Price Tier")
    st.plotly_chart(fig3, width="stretch")

    seg2 = filtered.groupby("fuel", as_index=False)["Units_Sold"].sum()
    fig4 = px.bar(seg2, x="fuel", y="Units_Sold", color="fuel", title="Fuel Preference Analysis")
    st.plotly_chart(fig4, width="stretch")

    seg3 = filtered.groupby(["Category","Price_Tier"], as_index=False)["Units_Sold"].sum()
    fig5 = px.treemap(seg3, path=["Category","Price_Tier"], values="Units_Sold", title="Category vs Price Tier Mix")
    st.plotly_chart(fig5, width="stretch")

# ---------------------------
# Drilldown
# ---------------------------
with tabs[3]:
    st.markdown('<div class="big-title">🔍 City Drilldown</div>', unsafe_allow_html=True)
    city_list = sorted(df["City"].unique())
    city_choice = st.selectbox("Choose a city", options=["-- Select city --"] + city_list)

    if city_choice != "-- Select city --":
        city_df = filtered[filtered["City"] == city_choice].copy()
        if city_df.empty:
            st.warning("No data for selected city (after filters).")
        else:
            st.markdown(f"### 📍 {city_choice.title()} — Summary")
            st.write(f"- Total units sold: **{int(city_df['Units_Sold'].sum()):,}**")
            st.write(f"- Total revenue: **{format_inr(city_df['Revenue'].sum())}**")
            st.write(f"- Average selling price: **{format_inr(city_df['selling_price'].mean())}**")
            st.write(f"- Most common fuel: **{city_df['fuel'].mode()[0] if not city_df['fuel'].mode().empty else 'N/A'}**")

            model_sales = city_df.groupby("Model_name", as_index=False).agg(
                Units_Sold=("Units_Sold","sum"),
                Revenue=("Revenue","sum"),
                Avg_Price=("selling_price","mean")
            )
            top3 = model_sales.sort_values("Units_Sold", ascending=False).head(3)
            least3 = model_sales[~model_sales["Model_name"].isin(top3["Model_name"])]
            least3 = least3.sort_values("Units_Sold", ascending=True).head(3)

            all_models = list(top3["Model_name"]) + list(least3["Model_name"])
            reasons_map = assign_reasons(all_models)

            st.markdown("#### 🔝 Top 3 Models")
            st.table(top3.assign(
                Reason=top3["Model_name"].map(reasons_map),
                Revenue=top3["Revenue"].apply(format_inr),
                Avg_Price=top3["Avg_Price"].apply(format_inr)
            ))

            st.markdown("#### ⬇️ Least 3 Models")
            st.table(least3.assign(
                Reason=least3["Model_name"].map(reasons_map),
                Revenue=least3["Revenue"].apply(format_inr),
                Avg_Price=least3["Avg_Price"].apply(format_inr)
            ))

            st.markdown("#### 📊 Units Sold vs Selling Price")
            fig = px.scatter(
                city_df,
                x="selling_price", y="Units_Sold",
                color="Category",
                hover_data=["Model_name","fuel"],
                title=f"Units Sold vs Price — {city_choice.title()}"
            )
            st.plotly_chart(fig, width="stretch")

            if "Date" in city_df.columns:
                try:
                    city_df["Year"] = city_df["Date"].dt.year
                    year_model_sales = city_df.groupby(["Year","Model_name"], as_index=False)["Units_Sold"].sum()
                    if not year_model_sales.empty:
                        st.markdown("#### 📈 Year vs Model Sold")
                        fig2 = px.bar(
                            year_model_sales,
                            x="Year", y="Units_Sold",
                            color="Model_name",
                            barmode="group",
                            title=f"Year-wise Model Sales — {city_choice.title()}"
                        )
                        st.plotly_chart(fig2, width="stretch")
                except Exception as e:
                    st.warning(f"Could not plot Year vs Model Sold: {e}")

# ---------------------------
# Extras
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Extras")
st.sidebar.download_button(
    label="Download full data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="vehicle_sales_full.csv",
    mime="text/csv"
)
sample = generate_sample_dataset