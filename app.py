import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer  # Added for better NaN handling
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Electricity Demand Analysis Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Electricity Demand Analysis Dashboard")
st.markdown("""
This dashboard provides interactive analysis of electricity demand patterns and weather correlations,
including clustering analysis and demand forecasting.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('merged_weather_demand_data.csv')
        # Convert date column to datetime if it exists
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                continue
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Enhanced preprocessing function with better NaN handling
def preprocess_data(df, selected_features=None):
    if selected_features is None:
        # Select numeric columns for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_features = numeric_cols.tolist()
    
    # Handle missing values using SimpleImputer (more robust than fillna)
    df_subset = df[selected_features].copy()
    
    # Check for NaN values
    nan_columns = df_subset.columns[df_subset.isna().any()].tolist()
    if nan_columns:
        st.info(f"Imputing missing values in columns: {', '.join(nan_columns)}")
        
    # Use SimpleImputer for handling NaN values
    imputer = SimpleImputer(strategy='mean')
    df_processed = pd.DataFrame(
        imputer.fit_transform(df_subset),
        columns=selected_features
    )
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_processed)
    
    return scaled_data, scaler, selected_features, df_processed

# Clustering function
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# PCA function for visualization
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    return pca_result, pca

# Function to check data quality
def check_data_quality(df, selected_features):
    issues = []
    
    # Check for missing values
    missing_values = df[selected_features].isna().sum()
    if missing_values.sum() > 0:
        for col in missing_values.index:
            if missing_values[col] > 0:
                issues.append(f"Column '{col}' has {missing_values[col]} missing values")
    
    # Check for infinite values
    inf_values = df[selected_features].isin([np.inf, -np.inf]).sum()
    if inf_values.sum() > 0:
        for col in inf_values.index:
            if inf_values[col] > 0:
                issues.append(f"Column '{col}' has {inf_values[col]} infinite values")
    
    return issues

# Time series forecasting function
def create_time_series_features(df, date_col, target_col, lookback_window):
    """Create time series features for forecasting"""
    # Sort by date
    df = df.sort_values(by=date_col)
    
    # Create lag features
    for lag in range(1, lookback_window + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Drop rows with NaN created by lag features
    df = df.dropna()
    
    return df

# Main content
try:
    # Load data
    df = load_data()
    
    if df is not None:
        # Input Form Section
        st.header("Input Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique cities if city column exists
            if 'city' in df.columns:
                cities = df['city'].unique()
                city = st.selectbox(
                    "Select City",
                    options=cities,
                    help="Choose the city for analysis"
                )
                df = df[df['city'] == city]
            
            # Date range selection
            date_col = None
            for potential_date_col in ['date', 'datetime', 'time', 'timestamp']:
                date_cols = [col for col in df.columns if potential_date_col in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    break
            
            if date_col:
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the start and end dates for analysis"
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df = df[(df[date_col].dt.date >= start_date) & 
                            (df[date_col].dt.date <= end_date)]
        
        with col2:
            lookback_window = st.slider(
                "Look-back Window",
                min_value=1,
                max_value=30,
                value=10,
                help="Number of previous days to consider for forecasting"
            )
            
            n_clusters = st.slider(
                "Number of Clusters (k)",
                min_value=2,
                max_value=10,
                value=4,
                help="Number of clusters for K-means clustering"
            )
        
        # Feature Selection
        st.subheader("Feature Selection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Pre-select some demand and weather related features if they exist
        default_features = []
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['demand', 'consumption', 'load', 'kwh', 'mwh']):
                default_features.append(col)
        
        # Add some weather features if available
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['temp', 'humidity', 'wind', 'precip']):
                if len(default_features) < 5:  # Limit to 5 default features
                    default_features.append(col)
        
        # If we couldn't find good defaults, use the first few numeric columns
        if not default_features and len(numeric_cols) > 0:
            default_features = numeric_cols[:min(4, len(numeric_cols))].tolist()
        
        selected_features = st.multiselect(
            "Select Features for Analysis",
            options=numeric_cols,
            default=default_features,
            help="Choose features for clustering and analysis"
        )
        
        if selected_features:
            # Check data quality and report issues
            issues = check_data_quality(df, selected_features)
            if issues:
                with st.expander("⚠️ Data Quality Issues Detected", expanded=True):
                    st.warning("The following issues were detected in your data:")
                    for issue in issues:
                        st.write(f"- {issue}")
                    st.write("These issues will be handled automatically through imputation.")
            
            # Preprocess data with enhanced NaN handling
            scaled_data, scaler, _, imputed_data = preprocess_data(df, selected_features)
            
            # Perform clustering
            clusters, kmeans = perform_clustering(scaled_data, n_clusters)
            
            # Add cluster information back to the dataframe
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = clusters
            
            # Perform PCA for visualization
            pca_result, pca = perform_pca(scaled_data)
            
            # Results Display Section
            st.header("Analysis Results")
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Cluster Analysis", "Feature Analysis", "Time Series", "Model Metrics"])
            
            with tab1:
                st.subheader("Cluster Visualization")
                # Create PCA plot
                pca_df = pd.DataFrame(
                    data=pca_result,
                    columns=['PC1', 'PC2']
                )
                pca_df['Cluster'] = clusters
                
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title='PCA Visualization of Clusters',
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                           'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster centers
                st.subheader("Cluster Centers")
                cluster_centers = pd.DataFrame(
                    scaler.inverse_transform(kmeans.cluster_centers_),
                    columns=selected_features
                )
                st.dataframe(cluster_centers, use_container_width=True)
                
                # Show cluster distribution
                st.subheader("Cluster Distribution")
                cluster_counts = df_with_clusters['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                fig = px.bar(
                    cluster_counts, 
                    x='Cluster', 
                    y='Count',
                    title='Number of Data Points in Each Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Feature Analysis")
                
                # Correlation matrix
                st.subheader("Feature Correlations")
                corr_matrix = imputed_data.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature distributions by cluster
                st.subheader("Feature Distributions by Cluster")
                selected_feature = st.selectbox(
                    "Select Feature to Visualize",
                    options=selected_features
                )
                
                fig = px.box(
                    df_with_clusters, 
                    x='Cluster', 
                    y=selected_feature,
                    title=f'Distribution of {selected_feature} by Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot matrix for selected features
                if len(selected_features) >= 2:
                    st.subheader("Feature Relationships")
                    scatter_features = st.multiselect(
                        "Select Features for Scatter Plot (2-4 recommended)",
                        options=selected_features,
                        default=selected_features[:min(3, len(selected_features))]
                    )
                    
                    if len(scatter_features) >= 2:
                        fig = px.scatter_matrix(
                            imputed_data[scatter_features],
                            dimensions=scatter_features,
                            title="Scatter Plot Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Time Series Analysis")
                
                # Find potential demand columns
                demand_cols = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['demand', 'consumption', 'load', 'kwh', 'mwh']
                )]
                
                if date_col and demand_cols:
                    target_col = st.selectbox(
                        "Select Demand/Consumption Column",
                        options=demand_cols,
                        index=0
                    )
                    
                    # Time series plot
                    fig = px.line(
                        df, 
                        x=date_col, 
                        y=target_col,
                        title=f'{target_col} Over Time'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Daily and weekly patterns
                    if len(df) > 24:  # Ensure enough data
                        st.subheader("Daily & Weekly Patterns")
                        
                        try:
                            # Extract hour and day of week
                            df['hour'] = df[date_col].dt.hour
                            df['day_of_week'] = df[date_col].dt.dayofweek
                            
                            # Daily pattern
                            daily_pattern = df.groupby('hour')[target_col].mean().reset_index()
                            fig = px.line(
                                daily_pattern,
                                x='hour',
                                y=target_col,
                                title=f'Average {target_col} by Hour of Day',
                                labels={'hour': 'Hour of Day', target_col: 'Average Value'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Weekly pattern
                            weekly_pattern = df.groupby('day_of_week')[target_col].mean().reset_index()
                            weekly_pattern['day_name'] = weekly_pattern['day_of_week'].map({
                                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                                4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                            })
                            fig = px.bar(
                                weekly_pattern,
                                x='day_name',
                                y=target_col,
                                title=f'Average {target_col} by Day of Week',
                                labels={target_col: 'Average Value'}
                            )
                            fig.update_xaxes(categoryorder='array', 
                                          categoryarray=['Monday', 'Tuesday', 'Wednesday', 
                                                         'Thursday', 'Friday', 'Saturday', 'Sunday'])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in temporal analysis: {str(e)}")
                
                else:
                    st.info("Time series analysis requires a date column and a demand/consumption column")
            
            with tab4:
                st.subheader("Model Performance Metrics")
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                try:
                    silhouette_avg = silhouette_score(scaled_data, clusters)
                    
                    # Basic cluster metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                        st.metric("Number of Clusters", n_clusters)
                    
                    with col2:
                        st.metric("Data Points", len(df))
                        st.metric("Number of Features", len(selected_features))
                    
                    # Inertia (within-cluster sum of squares)
                    st.metric("Inertia (Within-cluster Sum of Squares)", f"{kmeans.inertia_:.2f}")
                    
                    # PCA explained variance
                    st.subheader("PCA Explained Variance")
                    explained_variance = pca.explained_variance_ratio_
                    cum_explained_variance = np.cumsum(explained_variance)
                    
                    explained_var_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                        'Explained Variance': explained_variance,
                        'Cumulative Explained Variance': cum_explained_variance
                    })
                    
                    fig = px.bar(
                        explained_var_df,
                        x='Component',
                        y='Explained Variance',
                        title='Explained Variance by PCA Component'
                    )
                    fig.add_scatter(
                        x=explained_var_df['Component'],
                        y=explained_var_df['Cumulative Explained Variance'],
                        mode='lines+markers',
                        name='Cumulative'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")
        else:
            st.warning("Please select at least one feature for analysis.")
        
        # Help & Documentation Section
        st.header("Documentation")
        with st.expander("How to Use This Dashboard"):
            st.markdown("""
            ### Using the Dashboard
            
            1. **Input Parameters**:
               - Select your city of interest (if available)
               - Choose the date range for analysis
               - Adjust the look-back window and number of clusters
            
            2. **Feature Selection**:
               - Choose the features you want to analyze
               - Select at least 2 features for clustering
            
            3. **Results Interpretation**:
               - Cluster Analysis: View demand patterns grouped by similarity
               - Feature Analysis: Explore relationships between features
               - Time Series: Analyze temporal patterns in the data
               - Model Metrics: Evaluate clustering performance
            """)
        
        with st.expander("Technical Details"):
            st.markdown("""
            ### Technical Information
            
            **Data Processing**:
            - Missing values are handled using SimpleImputer with mean strategy
            - Features are standardized using StandardScaler
            - PCA is used for dimensionality reduction and visualization
            
            **Algorithms Used**:
            - K-means Clustering
            - Principal Component Analysis (PCA)
            - Correlation Analysis
            
            **Performance Metrics**:
            - Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters
            - Inertia: Sum of squared distances of samples to their closest cluster center
            - Explained variance ratio: How much information (variance) can be attributed to each principal component
            """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure all required data and models are properly loaded.")