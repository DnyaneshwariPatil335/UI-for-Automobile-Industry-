# main.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import numpy as np
import locale

warnings.filterwarnings("ignore")

# Set locale for Indian Rupee formatting
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, 'en_IN')

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="🚗 Vehicle Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------- Custom CSS with Sea Blue Theme and Pink File Upload ----------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0077be 0%, #00a0e4 50%, #87ceeb 100%);
        background-attachment: fixed;
    }
    
    /* Main content containers */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* All text black */
    body, p, div, span, label, input, select, textarea, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Headers with black color */
    h1, h2, h3, h4 {
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #005580 0%, #0077be 100%) !important;
        border-right: 2px solid #00a0e4;
    }
    
    /* Sidebar text - white for contrast */
    .css-1d391kg p, .css-1d391kg label, .css-1d391kg div, .css-1d391kg span {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        color: white !important;
        background: linear-gradient(45deg, #0077be, #00a0e4);
        border-radius: 12px;
        padding: 0.7em 1.5em;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 119, 190, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #005580, #0077be);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 119, 190, 0.4);
    }
    
    /* Metrics styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 248, 255, 0.9));
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(0, 119, 190, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0, 119, 190, 0.1), rgba(0, 160, 228, 0.1));
        border-radius: 8px;
        font-weight: 600;
        color: #000000 !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        border-left: 3px solid #00a0e4;
        color: #000000 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        color: #000000 !important;
    }
    
    /* Select box styling */
    .stSelectbox, .stMultiselect {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        color: #000000 !important;
    }
    
    /* Info box styling */
    .stInfo {
        background-color: rgba(240, 248, 255, 0.95) !important;
        border-left: 4px solid #0077be;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(230, 245, 253, 0.95) !important;
        border-left: 4px solid #00a0e4;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    /* Chart containers */
    .stPlot {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 极速加速器 0.05);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30极速加速器 px;
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Make sure all text in widgets is black */
    .stSelectbox label, .stMultiselect label, .stSlider label, .stRadio label {
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 8px 8极速加速器 px 0 0;
        padding: 10px 16px;
        color: #000000 !important;
        font-weight: 600;
极速加速器    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #000000 !important;
    }
    
    /* File uploader styling - CHANGED TO PINK */
    .stFileUploader label {
        color: #ff4b8b !important; /* Changed from black to pink */
        font-weight: 700;
        font-size: 18px;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.7);
    }
    
    /* File uploader button styling */
    .stFileUploader button {
        background: linear-gradient(45极速加速器 deg, #ff4b8b, #ff85a2) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stFileUploader button:hover {
        background: linear-gradient(45deg, #ff2d7a, #ff6b94) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 139, 0.4);
    }
    
    /* File uploader drag and drop area */
    .stFileUploader > div > div {
        border: 2px dashed #ff4b8b !important;
        border-radius: 10px;
        background-color: rgba(255, 245, 248, 0.9) !important;
    }
    
    /* File uploader text */
    .stFileUploader > div > div > div {
        color: #ff4b8b !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Title with Icon ----------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🚗 Vehicle Sales Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #000000 !important;'>Interactive <strong>Vehicle Sales Analysis</strong> with filters, insights, and advanced visualizations 📊</p>", unsafe_allow_html=True)

# ---------------------- File Upload ----------------------
with st.container():
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV/Excel)", type=["csv", "xlsx"], help="Upload your vehicle sales data to get started")

if uploaded_file is not None:
    # Save uploaded file
    save_dir = "uploaded_files"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File saved at `{save_path}`")

    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(save_path)
    else:
        df = pd.read_excel(save_path)

    # ✅ Always use Model_Name (fallback: rename Model -> Model_Name)
    if "Model_Name" not in df.columns and "Model" in df.columns:
        df.rename(columns={"Model": "Model_Name"}, inplace=True)

    # ---------------------- Dataset Preview ----------------------
    with st.expander("👀 Preview Uploaded Dataset", expanded=False):
        st.dataframe(df.head(20))

    # ---------------------- Sidebar Filters ----------------------
    st.sidebar.header("🔎 Filters")
    st.sidebar.markdown("---")
    df_filtered = df.copy()

    if "year" in df.columns:
        # Convert year to integer to ensure proper filtering
        df['year'] = df['year'].astype(int)
        years = st.sidebar.multiselect("📅 Select Year(s)", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
        df_filtered = df_filtered[ df_filtered['year'].isin(years)]

    if "Category" in df.columns:
        categories = st.sidebar.multiselect("🚗 Select Category", df['Category'].unique(), default=df['Category'].unique())
        df_filtered = df_filtered[df_filtered['Category'].isin(categories)]

    if "fuel" in df.columns:
        fuel_types = st.sidebar.multiselect("⛽ Select Fuel Type", df['fuel'].unique(), default=df['fuel'].unique())
        df_filtered = df_filtered[df_filtered['fuel'].isin(fuel_types)]

    if "seller_type" in df.columns:
        seller_types = st.sidebar.multiselect("🏪 Select Seller Type", df['seller_type'].unique(), default=df['seller_type'].unique())
        df_filtered = df_filtered[df_filtered['seller_type'].isin(seller_types)]

    # ✅ Model_Name filter (always clean names)
    if "Model_Name" in df.columns:
        # Sort models alphabetically for easier selection
        sorted_models = sorted(df['Model_Name'].unique())
        models = st.sidebar.multiselect("🚘 Select Model(s)", sorted_models, default=sorted_models)
        df_filtered = df_filtered[df_filtered['Model_Name'].isin(models)]

    # ---------------------- Metrics ----------------------
    st.markdown("---")
    st.header("📊 Key Performance Indicators")
    
    if "Units_Sold" in df_filtered.columns and "selling_price" in df_filtered.columns:
        df_filtered["revenue"] = df_filtered["Units_Sold"] * df_filtered["selling_price"]

        total_units = int(df_filtered["Units_Sold"].sum())
        total_revenue = float(df_filtered["revenue"].sum())
        avg_price = float(df_filtered["selling_price"].mean())
        total_models = len(df_filtered["Model_Name"].unique()) if "Model_Name" in df_filtered.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚗 Total Units Sold", f"{total_units:,}")
        
        # Format revenue and price in Indian Rupee format
        col2.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
        col3.metric("📊 Average Price", f"₹{avg_price:,.0f}")
        col4.metric("🔧 Unique Models", f"{total_models}")

    # ---------------------- Matplotlib Visualizations ----------------------
    st.markdown("---")
    st.header("📈 Data Visualizations")
    
    # Set style for plots with black text
    plt.style.use('default')
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.title极速加速器 color'] = 'black'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Format large numbers for better readability
    def format_large_number(num, pos=None):
        if num >= 1e6:
            return f'₹{num/1e6:.1f}M'
        elif num >= 1e3:
            return f'₹{num/1e3:.0f}K'
        else:
            return f'₹{num:.0f}'
    
    sns.set_palette("husl")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sales Analysis", "💰 Price Analysis", "🔗 Correlations", "📈 Trends", "🧮 Cross Analysis"])

    with tab1:
        # Sales by Category
        if "Category" in df_filtered.columns and "Units_Sold" in df_filtered.columns:
            st.subheader("Sales by Category")
            fig, ax = plt.subplots(figsize=(10, 6))
            category_data = df_filtered.groupby("Category")["Units_Sold"].sum().sort_values(ascending=False)
            bars = ax.bar(category_data.index, category_data.values, color=['#0077be', '#00a0e4', '#87ceeb', '#4682b4', '#5f9ea0'])
            ax.set_title("Total Sales by Vehicle Category", fontweight='bold', pad=20, color='black')
            ax.set_ylabel("Units Sold", color='black')
            ax.set_xlabel("Category", color='black')
            
            # Rotate x-axis labels by 45 degrees
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', color='black')
            
            st.pyplot(fig)

        # Top 10 Models by Sales
        if "Model_Name" in df_filtered.columns and "Units_Sold" in df_filtered.columns:
            st.subheader("Top 10 Models by Sales")
            model_data = df_filtered.groupby("Model_Name")["Units_Sold"].sum().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(model_data.index, model_data.values, color=['#0077be', '#00a0e4', '#87ceeb', '#4682b4', '#5f9ea0', '#6a5acd', '#7b68ee', '#9370db', '#8a2be2', '#9400d3'])
            ax.set_title("Top 10 Models by Sales Volume", fontweight='bold', pad=20, color='black')
            ax.set_xlabel("Units Sold", color='black')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{int(width):,}', ha='left', va='center', color='black')
            
            st.pyplot(fig)

    with tab2:
        # Boxplot
        if "Category" in df_filtered.columns and "selling_price" in df_filtered.columns:
            st.subheader("Price Distribution by Category")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Category", y="selling_price", data=df_filtered, ax=ax)
            ax.set_title("Selling Price Distribution by Category", fontweight='bold', pad=20, color='black')
            
            # Format y-axis with ₹ symbol and proper formatting
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
            
            ax.set_ylabel("Selling Price", color='black')
            ax.set_xlabel("Category", color='black')
            
            # Rotate x-axis labels by 45 degrees
            plt.xticks(rotation=45, ha='right')
            
            st.pyplot(fig)

        # Distribution Plot - FIXED: Rotate x-axis labels by 45 degrees
        if "selling_price" in df_filtered.columns:
            st.subheader("Price Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df_filtered["selling_price"], kde=True, color="#0077be", ax极速加速器=ax)
            ax.set_title("Distribution of Selling Prices", fontweight='bold', pad=20, color='black')
            
            # Format x-axis with ₹ symbol and rotate labels by 45 degrees
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
            
            # Rotate x-axis labels by 45 degrees
            plt.xticks(rotation=45, ha='right')
            
            ax.set_xlabel("Selling Price", color='black')
            ax.set_ylabel("Count", color='black')
            st.pyplot(fig)

    with tab3:
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = df_filtered.select_dtypes(include="number")
        if not numeric_cols.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, center=0)
            ax.set_title("Feature Correlation Matrix", fontweight='bold', pad=20, color='black')
            
            # Rotate x-axis labels by 45 degrees
            plt.xticks(rotation=45, ha='right')
            
            st.pyplot(fig)

    with tab4:
        # Sales by Year - FIXED VERSION
        if "year" in df_filtered.columns and "Units_Sold" in df_filtered.columns:
            st.subheader("Sales Trend Over Years")
            year_data = df_filtered.groupby("year")["Units_Sold"].sum()
            
            # Ensure years are integers and sort them
            year_data.index = year_data.index.astype(int)
            year_data = year_data.sort_index()
            
            # Filter to only include years 2019-2023 if they exist in the data
            valid_years = [year for year in [2019, 2020, 2021, 2022, 2023] if year in year_data.index]
            year_data = year_data.loc[valid_years]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(year_data.index, year_data.values, marker='o', linewidth=2, markersize=8, color='#0077be')
            ax.set_title("Sales Trend Over Years", fontweight='bold', pad=20, color='black')
            ax.set_ylabel("Units Sold", color='black')
            ax.set_xlabel("Year", color='black')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to only show integer years
            ax.set_xticks(year_data.index)
            ax.set_xticklabels(year_data.index.astype(int))
            
            # Add value labels on points
            for i, v in enumerate(year_data.values):
                ax.text(year_data.index[i], v + 0.1, f'{int(v):,}', ha='center', va='bottom', color='black')
            
            st.pyplot(fig)

        # Sales by Fuel Type
        if "fuel" in df_filtered.columns and "Units_Sold" in df_filtered.columns:
            st.subheader("Sales by Fuel Type")
            fuel_data = df_filtered.groupby("fuel")["Units_Sold"].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(fuel_data.index, fuel_data.values, color=['#0077be', '#00a0e4', '#87ceeb', '#4682b4', '#5f9ea0'])
            ax.set_title("Sales by Fuel Type", fontweight='bold', pad=20, color='black')
            ax.set_ylabel("Units Sold", color='black')
            ax.set_xlabel("Fuel Type", color='black')
            
            # Rotate x-axis labels by 45 degrees
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', color='black')
            
            st.pyplot(fig)

    with tab5:
        # Crosstab Analysis
        st.subheader("Category vs Fuel Type Analysis")
        if "Category" in df_filtered.columns and "fuel" in df_filtered.columns:
            ct = pd.crosstab(df_filtered["Category"], df_filtered["fuel"])
            st.dataframe(ct.style.background_gradient(cmap='Blues'))

        # Scatterplot
        if "Units_Sold" in df_filtered.columns and "selling_price" in df_filtered.columns:
            st.subheader("Units Sold vs Selling Price")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x="Units_Sold", y="selling_price", data=df_filtered, hue="Category", ax=ax, alpha=0.7, s=100)
            ax.set_title("Units Sold vs Selling Price by Category", fontweight='bold', pad=20, color='black')
            ax.set_xlabel("Units Sold", color='black')
            
            # Format y-axis with ₹ symbol
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
            ax.set_ylabel("Selling Price", color='black')
            
            ax.legend(title='Category', title_fontsize='10', fontsize='9')
            st.pyplot(fig)

        # Model-wise analysis
        if "Model_Name" in df_filtered.columns and "Units_Sold" in df_filtered.columns and "selling_price" in df_filtered.columns:
            st.subheader("Top Models Analysis")
            model_analysis = df_filtered.groupby("Model_Name").agg({
                "Units_Sold": "sum",
                "selling_price": "mean"
            }).sort_values("Units_Sold", ascending=False).head(10)
            
            # Format the selling price column with ₹ symbol
            formatted_analysis = model_analysis.copy()
            formatted_analysis["selling_price"] = formatted_analysis["selling_price"].apply(lambda x: f"₹{x:,.0f}")
            formatted_analysis["Units_Sold"] = formatted_analysis["Units_Sold"].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(formatted_analysis.style.background_gradient(cmap='Blues', subset=pd.IndexSlice[:, ['Units_Sold']]))

else:
    st.info("👆 Upload a dataset to start analysis")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>🚗 Vehicle Sales Analytics Dashboard | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)