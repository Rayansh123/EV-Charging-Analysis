import streamlit as st
import pandas as pd
import altair as alt
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# --- Page Config ---
st.set_page_config(
    page_title="EV Charging Analysis",
    page_icon="⚡",
    layout="wide"
)

# --- Caching ---
# We cache the data so it's only loaded once
@st.cache_data
def load_csv(file_path):
    # Spark saves CSVs with headers in a "part-0000" file
    # This is a robust way to find and load it
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# We cache the model so it's only loaded once
@st.cache_resource
def load_spark_model():
    """
    Initializes a local SparkSession and loads the saved ML model.
    """
    # Create a local SparkSession
    # This is necessary to load and run the PySpark model
    try:
        spark = (SparkSession.builder
                 .appName("StreamlitApp")
                 .master("local[1]")  # Use 1 core
                 .config("spark.driver.memory", "1g") # Limit memory
                 .getOrCreate())
    except Exception as e:
        st.error(f"Error creating Spark session: {e}")
        return None, None

    # Load the trained model pipeline
    try:
        model = PipelineModel.load('dt_model_pipeline')
        return spark, model
    except Exception as e:
        st.error(f"Error loading model from 'dt_model_pipeline': {e}")
        st.info("Ensure the 'dt_model_pipeline' folder is in the root directory.")
        return spark, None

# --- Load All Data ---
peak_hours_df = load_csv('app_data/peak_hours_overall.csv')
weekday_weekend_df = load_csv('app_data/peak_hours_weekday_vs_weekend.csv')
cluster_summary_df = load_csv('app_data/cluster_summary.csv')

# --- Load Spark Model ---
spark, model = load_spark_model()

# --- Sidebar Navigation ---
st.sidebar.title("⚡ EV Charging Analysis")
page = st.sidebar.radio(
    "Navigate Your Project",
    ["Welcome", "1. Exploratory Data Analysis", "2. Energy Prediction Model"]
)

# ==============================================================================
# PAGE: WELCOME
# ==============================================================================
if page == "Welcome":
    st.title("Big Data Capstone Project: EV Charging Analysis")
    st.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/images/sei_blog_post_header_image.max-1000x1000.png")
    
    st.header("Project Overview")
    st.write("""
    This app is a frontend for a Big Data Analytics project. The goal is to analyze
    large-scale Electric Vehicle (EV) charging session data to identify usage patterns
    and predict energy consumption.

    All data was processed on a **Google Cloud Dataproc** cluster using **PySpark**.
    """)

    st.subheader("Technology Stack")
    st.markdown("""
    - **Cloud & Storage:** Google Cloud Platform (GCP)
    - **Big Data Cluster:** Google Cloud Dataproc
    - **Data Processing:** Apache Spark (PySpark)
    - **Machine Learning:** PySpark.ml
    - **Frontend:** Streamlit
    - **Deployment:** GitHub & Streamlit Cloud
    """)

    st.subheader("Use the sidebar to navigate the project.")

# ==============================================================================
# PAGE: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page == "1. Exploratory Data Analysis":
    st.title("1. Exploratory Data Analysis (EDA)")
    
    st.header("Charger Type Clusters (K-Means)")
    st.write("""
    We used K-Means clustering on the 'charging rate (kW)' to find 3 distinct types of charging sessions.
    The results clearly show 'Slow', 'Medium', and 'Fast' chargers:
    """)
    if not cluster_summary_df.empty:
        st.dataframe(cluster_summary_df, use_container_width=True)
    else:
        st.warning("Cluster summary data not loaded.")
        
    st.header("Peak Charging Hours")
    st.write("We analyzed the most popular times for charging sessions to begin.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Peak Hours")
        if not peak_hours_df.empty:
            chart_peak = alt.Chart(peak_hours_df).mark_bar().encode(
                x=alt.X('hour_of_day:O', title='Hour of the Day'),
                y=alt.Y('count', title='Total Sessions'),
                tooltip=['hour_of_day', 'count']
            ).properties(
                title="Peak Charging Start Hours (All Days)"
            ).interactive()
            st.altair_chart(chart_peak, use_container_width=True)
        else:
            st.warning("Peak hours data not loaded.")

    with col2:
        st.subheader("Weekday vs. Weekend")
        if not weekday_weekend_df.empty:
            # Map boolean to a readable string for the legend
            source = weekday_weekend_df.copy()
            source['Day Type'] = source['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
            
            chart_wknd = alt.Chart(source).mark_line(point=True).encode(
                x=alt.X('hour_of_day:O', title='Hour of the Day'),
                y=alt.Y('count', title='Total Sessions'),
                color=alt.Color('Day Type', title='Day Type'),
                tooltip=['hour_of_day', 'count', 'Day Type']
            ).properties(
                title="Weekday vs. Weekend Charging Patterns"
            ).interactive()
            st.altair_chart(chart_wknd, use_container_width=True)
        else:
            st.warning("Weekday/weekend data not loaded.")

# ==============================================================================
# PAGE: PREDICTION MODEL
# ==============================================================================
elif page == "2. Energy Prediction Model":
    st.title("2. Predict Energy (kWh) Delivered")
    st.write("""
    We trained a **Decision Tree Regressor** model to predict the total energy (kWh)
    a session will deliver based on its properties. Use the controls on the left
    to see a live prediction!
    """)

    if model is None or spark is None:
        st.error("Model or Spark Session failed to load. Please check logs.")
    else:
        # --- User Input in Sidebar ---
        st.sidebar.header("Prediction Inputs")
        
        # User-friendly mapping for clusters
        # Based on your K-Means results:
        # 1: Slow (2.49 kW)
        # 0: Medium (7.12 kW)
        # 2: Fast (25.00 kW)
        cluster_map = {
            "Slow (L2 Home/Work, ~2.5 kW)": 1,
            "Medium (L2 Public, ~7.1 kW)": 0,
            "Fast (DC Fast Charger, ~25 kW)": 2
        }
        cluster_name = st.sidebar.selectbox(
            "Select Charger Type (from K-Means)",
            options=list(cluster_map.keys())
        )
        
        duration = st.sidebar.number_input(
            "Charging Duration (Hours)", 
            min_value=0.1, max_value=40.0, value=2.5, step=0.1
        )
        
        hour = st.sidebar.slider(
            "Hour of Day (Connection Time)", 
            min_value=0, max_value=23, value=16
        )
        
        # User-friendly mapping for day of week
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        day_name = st.sidebar.selectbox(
            "Day of Week",
            options=list(day_map.keys())
        )

        # --- Process Inputs & Predict ---
        
        # Convert user-friendly inputs back to model's numerical features
        day_of_week_num = day_map[day_name]
        is_weekend_bool = True if day_of_week_num in [5, 6] else False
        cluster_num = cluster_map[cluster_name]
        
        # 1. Create a single-row Pandas DataFrame
        # MUST match the schema used to train the VectorAssembler
        input_features = pd.DataFrame(
            [{
                "chargingDuration": duration,
                "hour_of_day": float(hour), # Ensure float/double
                "day_of_week": float(day_of_week_num),
                "is_weekend": is_weekend_bool,
                "prediction": float(cluster_num) # This was the cluster ID
            }]
        )
        
        # 2. Convert Pandas DataFrame to Spark DataFrame
        spark_df = spark.createDataFrame(input_features)
        
        # 3. Use the model pipeline to transform and predict
        # This runs the VectorAssembler and the Decision Tree
        prediction_result = model.transform(spark_df)
        
        # 4. Get the result
        predicted_kwh = prediction_result.collect()[0]['energy_prediction']
        
        # --- Display Results ---
        st.header("Prediction Result")
        st.metric(
            label="Predicted Energy Delivered",
            value=f"{predicted_kwh:.2f} kWh"
        )
        
        st.subheader("Model Inputs (Processed)")
        st.write("This is the raw data fed into the PySpark model:")
        st.dataframe(input_features, use_container_width=True)