# Part 1: Core Setup and Basic Functions

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(uploaded_file)
    
    # Convert numeric columns
    numeric_columns = [
        'Price', 'Monthly Sales', 'Monthly Revenue', 'Review Count',
        'Reviews Rating', 'Sales Trend (90 days) (%)', 'Price Trend (90 days) (%)'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^\d.-]', ''), errors='coerce')
    
    # Add derived metrics
    df['Revenue_per_Review'] = df['Monthly Revenue'] / df['Review Count']
    df['Sales_Efficiency'] = df['Monthly Sales'] / df['Price']
    df['Review_Rating_Normalized'] = (df['Reviews Rating'] - df['Reviews Rating'].mean()) / df['Reviews Rating'].std()
    
    return df

def calculate_basic_metrics(df):
    """Calculate basic market metrics"""
    metrics = {
        'Total_Products': len(df),
        'Total_Revenue': df['Monthly Revenue'].sum(),
        'Average_Price': df['Price'].mean(),
        'Average_Rating': df['Reviews Rating'].mean(),
        'Total_Sales': df['Monthly Sales'].sum(),
        'Total_Reviews': df['Review Count'].sum()
    }
    return metrics

def calculate_market_share(df):
    """Calculate market share by brand"""
    market_share = df.groupby('Brand')['Monthly Revenue'].sum()
    return (market_share / market_share.sum() * 100).round(2)

def calculate_growth_metrics(df):
    """Calculate growth metrics by brand"""
    return df.groupby('Brand').agg({
        'Sales Trend (90 days) (%)': 'mean',
        'Price Trend (90 days) (%)': 'mean'
    }).round(2)

@st.cache_data
def generate_color_palette(n_colors):
    """Generate a color palette for consistent visualization"""
    colors = px.colors.qualitative.Set3[:n_colors]
    return colors

class DataAnalyzer:
    """Main class for data analysis"""
    def __init__(self, df):
        self.df = df
        self.metrics = calculate_basic_metrics(df)
        self.market_share = calculate_market_share(df)
        self.growth_metrics = calculate_growth_metrics(df)
    
    def get_top_performers(self, metric, n=5):
        """Get top performing brands by metric"""
        return self.df.groupby('Brand')[metric].sum().nlargest(n)
    
    def get_price_segments(self):
        """Create price segments"""
        return pd.qcut(
            self.df['Price'],
            q=5,
            labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury']
        )
    
    def get_brand_metrics(self):
        """Calculate comprehensive brand metrics"""
        return self.df.groupby('Brand').agg({
            'Price': ['mean', 'min', 'max'],
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean',
            'Review Count': 'sum',
            'Sales Trend (90 days) (%)': 'mean'
        }).round(2)

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Stroller Market Analysis",
        page_icon="🛵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert > div {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def create_filters_sidebar(df):
    """Create sidebar filters"""
    st.sidebar.title("Analysis Filters")
    
    # Category filters
    categories = ['All Categories'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    if selected_category != 'All Categories':
        subcategories = ['All Subcategories'] + sorted(
            df[df['Category'] == selected_category]['Subcategory'].unique().tolist()
        )
    else:
        subcategories = ['All Subcategories'] + sorted(df['Subcategory'].unique().tolist())
    
    selected_subcategory = st.sidebar.selectbox("Select Subcategory", subcategories)
    
    # Price range
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        float(df['Price'].min()),
        float(df['Price'].max()),
        (float(df['Price'].min()), float(df['Price'].max()))
    )
    
    # Seller selection
    sellers = sorted(df['Seller'].unique())
    selected_seller = st.sidebar.selectbox("Select Seller", ['All Sellers'] + sellers)
    
    # Filter brands based on selected seller
    if selected_seller == 'All Sellers':
        available_brands = sorted(df['Brand'].unique())
    else:
        available_brands = sorted(df[df['Seller'] == selected_seller]['Brand'].unique())
    
    # Brand selection with "Select All" option
    select_all_brands = st.sidebar.checkbox("Select All Brands", value=False)
    
    if select_all_brands:
        selected_brands = st.sidebar.multiselect("Select Brands", available_brands, default=available_brands)
    else:
        selected_brands = st.sidebar.multiselect("Select Brands", available_brands)

    return {
        'category': selected_category,
        'subcategory': selected_subcategory,
        'price_range': price_range,
        'seller': selected_seller,
        'brands': selected_brands
    }

def apply_filters(df, filters):
    """Apply selected filters to dataframe"""
    filtered_df = df.copy()
    
    # Apply category filters
    if filters['category'] != 'All Categories':
        filtered_df = filtered_df[filtered_df['Category'] == filters['category']]
    if filters['subcategory'] != 'All Subcategories':
        filtered_df = filtered_df[filtered_df['Subcategory'] == filters['subcategory']]
    
    # Apply price and brand filters
    filtered_df = filtered_df[
        (filtered_df['Price'].between(filters['price_range'][0], filters['price_range'][1])) &
        (filtered_df['Brand'].isin(filters['brands']))
    ]
    
    return filtered_df

if __name__ == "__main__":
    # Initialize application
    setup_page_config()
    st.title("🛵 Baby Stroller Market Analysis System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        df = load_and_preprocess_data(uploaded_file)
        analyzer = DataAnalyzer(df)
        
        # Create filters
        filters = create_filters_sidebar(df)
        filtered_df = apply_filters(df, filters)
        
        # Store data in session state
        st.session_state['data'] = filtered_df
        st.session_state['analyzer'] = analyzer
        
        # Continue with analysis...

        # Part 2: Market Analysis & Visualization

class MarketAnalyzer:
    def __init__(self, df):
        self.df = df
        self.plots = {}

    def create_market_overview(self):
        """Generate market overview visualizations"""
        st.header("Market Overview")
        
        # Key Performance Indicators
        self.show_kpi_metrics()
        
        # Market Share Analysis
        self.create_market_share_analysis()
        
        # Price Segment Analysis
        self.create_price_segment_analysis()
        
        # Brand Performance Overview
        self.create_brand_performance_overview()
        
        return self.plots

    def show_kpi_metrics(self):
        """Display key performance indicators"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Products",
                f"{len(self.df):,}",
                f"{self.df['Sales Trend (90 days) (%)'].mean():+.1f}% Growth"
            )
        
        with col2:
            st.metric(
                "Monthly Revenue",
                f"₹{self.df['Monthly Revenue'].sum():,.0f}",
                f"{self.df['Price Trend (90 days) (%)'].mean():+.1f}% Trend"
            )
        
        with col3:
            st.metric(
                "Average Rating",
                f"{self.df['Reviews Rating'].mean():.2f}",
                f"{len(self.df[self.df['Reviews Rating'] >= 4.0])} Top Rated"
            )
        
        with col4:
            st.metric(
                "Total Sales",
                f"{self.df['Monthly Sales'].sum():,.0f}",
                f"{self.df['Sales Trend (90 days) (%)'].mean():+.1f}% Growth"
            )

    def create_market_share_analysis(self):
        """Create market share visualizations"""
        st.subheader("Market Share Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue Share
            fig_revenue = px.pie(
                self.df,
                values='Monthly Revenue',
                names='Brand',
                title='Revenue Share by Brand',
                hole=0.4
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
            self.plots['revenue_share'] = fig_revenue
        
        with col2:
            # Sales Volume Share
            fig_sales = px.pie(
                self.df,
                values='Monthly Sales',
                names='Brand',
                title='Sales Volume Share by Brand',
                hole=0.4
            )
            st.plotly_chart(fig_sales, use_container_width=True)
            self.plots['sales_share'] = fig_sales

    def create_price_segment_analysis(self):
        """Create price segment analysis"""
        st.subheader("Price Segment Analysis")
        
        # Create price segments
        self.df['Price_Segment'] = pd.qcut(
            self.df['Price'],
            q=5,
            labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury']
        )
        
        # Segment Distribution
        fig_segment = px.treemap(
            self.df,
            path=['Price_Segment', 'Brand'],
            values='Monthly Sales',
            color='Reviews Rating',
            title='Market Segmentation'
        )
        st.plotly_chart(fig_segment, use_container_width=True)
        self.plots['price_segments'] = fig_segment
        
        # Segment Performance Metrics
        segment_metrics = self.df.groupby('Price_Segment').agg({
            'Price': ['mean', 'min', 'max'],
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean'
        }).round(2)
        
        st.write("Segment Performance Metrics:")
        st.dataframe(segment_metrics)

    def create_brand_performance_overview(self):
        """Create brand performance overview"""
        st.subheader("Brand Performance Overview")
        
        # Calculate brand metrics
        brand_metrics = self.df.groupby('Brand').agg({
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean',
            'Price': 'mean',
            'Sales Trend (90 days) (%)': 'mean'
        }).round(2)
        
        # Create performance matrix
        fig_performance = px.scatter(
            brand_metrics.reset_index(),
            x='Price',
            y='Monthly Sales',
            size='Monthly Revenue',
            color='Reviews Rating',
            hover_data=['Sales Trend (90 days) (%)'],
            text='Brand',
            title='Brand Performance Matrix'
        )
        st.plotly_chart(fig_performance, use_container_width=True)
        self.plots['brand_performance'] = fig_performance

    def create_trend_analysis(self):
        """Create trend analysis visualizations"""
        st.subheader("Market Trend Analysis")
        
        # Sales Trend Distribution
        fig_trend = px.histogram(
            self.df,
            x='Sales Trend (90 days) (%)',
            color='Brand',
            title='Distribution of Sales Trends'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        self.plots['sales_trend'] = fig_trend
        
        # Price vs Sales Trend
        fig_price_trend = px.scatter(
            self.df,
            x='Price',
            y='Sales Trend (90 days) (%)',
            color='Brand',
            size='Monthly Sales',
            title='Price vs Sales Trend'
        )
        st.plotly_chart(fig_price_trend, use_container_width=True)
        self.plots['price_trend'] = fig_price_trend

    def create_geographic_analysis(self):
        """Create geographic analysis if location data is available"""
        if 'Location' in self.df.columns:
            st.subheader("Geographic Analysis")
            
            # Sales by location
            location_metrics = self.df.groupby('Location').agg({
                'Monthly Sales': 'sum',
                'Monthly Revenue': 'sum',
                'Reviews Rating': 'mean'
            }).round(2)
            
            # Create map visualization if coordinates are available
            if 'Latitude' in self.df.columns and 'Longitude' in self.df.columns:
                fig_map = px.scatter_mapbox(
                    self.df,
                    lat='Latitude',
                    lon='Longitude',
                    size='Monthly Sales',
                    color='Brand',
                    hover_data=['Monthly Revenue', 'Reviews Rating'],
                    title='Sales Distribution by Location'
                )
                st.plotly_chart(fig_map, use_container_width=True)
                self.plots['geographic'] = fig_map

    def generate_market_insights(self):
        """Generate key market insights"""
        st.subheader("Key Market Insights")
        
        insights = []
        
        # Market Leader Analysis
        market_leader = self.df.groupby('Brand')['Monthly Revenue'].sum().idxmax()
        market_share = (self.df[self.df['Brand'] == market_leader]['Monthly Revenue'].sum() / 
                       self.df['Monthly Revenue'].sum() * 100)
        
        insights.append(f"Market Leader: {market_leader} with {market_share:.1f}% market share")
        
        # Growth Analysis
        growth_leader = self.df.groupby('Brand')['Sales Trend (90 days) (%)'].mean().idxmax()
        growth_rate = self.df.groupby('Brand')['Sales Trend (90 days) (%)'].mean().max()
        
        insights.append(f"Fastest Growing Brand: {growth_leader} ({growth_rate:.1f}% growth)")
        
        # Price Analysis
        avg_price = self.df['Price'].mean()
        price_trend = self.df['Price Trend (90 days) (%)'].mean()
        
        insights.append(
            f"Average Market Price: ₹{avg_price:,.2f} with {price_trend:+.1f}% trend"
        )
        
        # Display insights
        for insight in insights:
            st.markdown(f"• {insight}")

def add_market_analysis_section():
    """Add market analysis section to the app"""
    if 'data' in st.session_state:
        analyzer = MarketAnalyzer(st.session_state['data'])
        
        # Analysis Type Selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Market Overview", "Trend Analysis", "Geographic Analysis", "Custom Analysis"]
        )
        
        if analysis_type == "Market Overview":
            plots = analyzer.create_market_overview()
        
        elif analysis_type == "Trend Analysis":
            analyzer.create_trend_analysis()
        
        elif analysis_type == "Geographic Analysis":
            analyzer.create_geographic_analysis()
        
        elif analysis_type == "Custom Analysis":
            st.subheader("Custom Market Analysis")
            
            # Let user select metrics
            x_axis = st.selectbox("Select X-axis metric", st.session_state['data'].columns)
            y_axis = st.selectbox("Select Y-axis metric", st.session_state['data'].columns)
            
            # Create custom visualization
            fig_custom = px.scatter(
                st.session_state['data'],
                x=x_axis,
                y=y_axis,
                color='Brand',
                size='Monthly Sales',
                title=f'{y_axis} vs {x_axis}'
            )
            st.plotly_chart(fig_custom, use_container_width=True)
        
        # Generate insights
        analyzer.generate_market_insights()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_market_analysis_section()

# Part 3: Statistical Analysis & Forecasting

class StatisticalAnalyzer:
    def __init__(self, df):
        self.df = df
        self.plots = {}
        self.time_series = self._prepare_time_series()

    def _prepare_time_series(self):
        """Prepare time series data"""
        df_time = self.df.copy()
        df_time['Date'] = pd.to_datetime(df_time['Best Sales Period'])
        return df_time.groupby('Date')['Monthly Sales'].sum().reset_index().set_index('Date')

    def run_statistical_analysis(self):
        """Run comprehensive statistical analysis"""
        st.header("Statistical Analysis & Forecasting")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Distribution Analysis", "Correlation Analysis", "Time Series Analysis", 
             "Forecasting", "Hypothesis Testing", "Advanced Metrics"]
        )
        
        if analysis_type == "Distribution Analysis":
            self.create_distribution_analysis()
        elif analysis_type == "Correlation Analysis":
            self.create_correlation_analysis()
        elif analysis_type == "Time Series Analysis":
            self.create_time_series_analysis()
        elif analysis_type == "Forecasting":
            self.create_sales_forecast()
        elif analysis_type == "Hypothesis Testing":
            self.perform_hypothesis_tests()
        elif analysis_type == "Advanced Metrics":
            self.show_advanced_metrics()

    def create_distribution_analysis(self):
        """Analyze distributions of key metrics"""
        st.subheader("Distribution Analysis")
        
        # Select metric for analysis
        metric = st.selectbox(
            "Select Metric",
            ["Price", "Monthly Sales", "Reviews Rating", "Review Count"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig_dist = ff.create_distplot(
                [self.df[metric].dropna()],
                [metric],
                show_hist=True,
                show_rug=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            self.plots['distribution'] = fig_dist
        
        with col2:
            # Basic statistics
            stats = self.df[metric].describe()
            st.write("Statistical Summary:")
            st.dataframe(stats)
            
            # Normality test
            stat, p_value = normaltest(self.df[metric].dropna().values)
            st.write("Normality Test:")
            st.write(f"p-value: {p_value:.4f}")
            st.write(f"Distribution is {'normally' if p_value > 0.05 else 'not normally'} distributed")

    def create_correlation_analysis(self):
        """Analyze correlations between metrics"""
        st.subheader("Correlation Analysis")
        
        # Select metrics for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        selected_metrics = st.multiselect(
            "Select Metrics for Correlation Analysis",
            numeric_cols,
            default=["Price", "Monthly Sales", "Reviews Rating"]
        )
        
        if len(selected_metrics) > 1:
            # Correlation matrix
            correlation_matrix = self.df[selected_metrics].corr()
            
            # Heatmap
            fig_corr = px.imshow(
                correlation_matrix,
                title="Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            self.plots['correlation'] = fig_corr
            
            # Detailed correlation analysis
            st.write("Significant Correlations:")
            for i in range(len(selected_metrics)):
                for j in range(i+1, len(selected_metrics)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.3:  # Show only meaningful correlations
                        st.write(f"{selected_metrics[i]} vs {selected_metrics[j]}: {corr:.3f}")

    def create_time_series_analysis(self):
        """Perform time series analysis"""
        st.subheader("Time Series Analysis")
        
        # Decomposition
        try:
            decomposition = seasonal_decompose(
                self.time_series['Monthly Sales'],
                period=12,
                extrapolate_trend='freq'
            )
            
            # Plot components
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
            )
            
            components = [
                ('Observed', decomposition.observed),
                ('Trend', decomposition.trend),
                ('Seasonal', decomposition.seasonal),
                ('Residual', decomposition.resid)
            ]
            
            for i, (name, data) in enumerate(components, 1):
                fig.add_trace(
                    go.Scatter(x=self.time_series.index, y=data, name=name),
                    row=i, col=1
                )
            
            fig.update_layout(height=800, title_text="Time Series Decomposition")
            st.plotly_chart(fig, use_container_width=True)
            self.plots['decomposition'] = fig
            
        except Exception as e:
            st.error(f"Error in time series decomposition: {str(e)}")

    def create_sales_forecast(self):
        """Create sales forecast"""
        st.subheader("Sales Forecasting")
        
        # Forecast parameters
        forecast_periods = st.slider("Forecast Periods (months)", 1, 12, 6)
        
        try:
            # Fit model
            model = ExponentialSmoothing(
                self.time_series['Monthly Sales'],
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(forecast_periods)
            
            # Create forecast plot
            fig_forecast = go.Figure()
            
            # Add actual values
            fig_forecast.add_trace(
                go.Scatter(
                    x=self.time_series.index,
                    y=self.time_series['Monthly Sales'],
                    name='Actual',
                    line=dict(color='blue')
                )
            )
            
            # Add forecast
            fig_forecast.add_trace(
                go.Scatter(
                    x=pd.date_range(
                        start=self.time_series.index[-1],
                        periods=forecast_periods+1,
                        freq='M'
                    )[1:],
                    y=forecast,
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig_forecast.update_layout(title='Sales Forecast')
            st.plotly_chart(fig_forecast, use_container_width=True)
            self.plots['forecast'] = fig_forecast
            
            # Show forecast metrics
            forecast_metrics = pd.DataFrame({
                'Forecasted Sales': forecast
            })
            st.write("Forecast Values:")
            st.dataframe(forecast_metrics)
            
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")

    def perform_hypothesis_tests(self):
        """Perform statistical hypothesis tests"""
        st.subheader("Hypothesis Testing")
        
        test_type = st.selectbox(
            "Select Test Type",
            ["ANOVA", "T-Test", "Chi-Square Test"]
        )
        
        if test_type == "ANOVA":
            # One-way ANOVA for brand comparison
            metric = st.selectbox("Select Metric", ["Price", "Monthly Sales", "Reviews Rating"])
            
            try:
                groups = [group[metric].values for name, group in self.df.groupby('Brand')]
                f_stat, p_val = stats.f_oneway(*groups)
                
                st.write("One-way ANOVA Results:")
                st.write(f"F-statistic: {f_stat:.4f}")
                st.write(f"p-value: {p_val:.4f}")
                st.write(f"{'Significant' if p_val < 0.05 else 'No significant'} difference between brands")
                
            except Exception as e:
                st.error(f"Error in ANOVA test: {str(e)}")

    def show_advanced_metrics(self):
        """Display advanced statistical metrics"""
        st.subheader("Advanced Statistical Metrics")
        
        metrics = ["Price", "Monthly Sales", "Reviews Rating"]
        advanced_stats = {}
        
        for metric in metrics:
            advanced_stats[metric] = {
                'Mean': self.df[metric].mean(),
                'Median': self.df[metric].median(),
                'Std Dev': self.df[metric].std(),
                'Skewness': self.df[metric].skew(),
                'Kurtosis': self.df[metric].kurtosis(),
                'CV (%)': (self.df[metric].std() / self.df[metric].mean()) * 100
            }
        
        # Display metrics
        st.dataframe(pd.DataFrame(advanced_stats))

def add_statistical_analysis_section():
    """Add statistical analysis section to the app"""
    if 'data' in st.session_state:
        analyzer = StatisticalAnalyzer(st.session_state['data'])
        analyzer.run_statistical_analysis()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_statistical_analysis_section()

# Part 4: Competitive Analysis and Brand Positioning

class CompetitiveAnalyzer:
    def __init__(self, df):
        self.df = df
        self.plots = {}
        self.metrics = self._calculate_competitive_metrics()

    def _calculate_competitive_metrics(self):
        """Calculate comprehensive competitive metrics"""
        metrics = {}
        
        # Brand performance metrics
        metrics['brand_performance'] = self.df.groupby('Brand').agg({
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean',
            'Price': ['mean', 'min', 'max'],
            'Sales Trend (90 days) (%)': 'mean'
        }).round(2)
        
        # Market share
        total_revenue = self.df['Monthly Revenue'].sum()
        metrics['market_share'] = (
            self.df.groupby('Brand')['Monthly Revenue'].sum() / total_revenue * 100
        ).round(2)
        
        # Competitive strength index
        metrics['strength_index'] = self._calculate_strength_index()
        
        return metrics

    def _calculate_strength_index(self):
        """Calculate competitive strength index for each brand"""
        brand_metrics = self.df.groupby('Brand').agg({
            'Price': 'mean',
            'Monthly Sales': 'sum',
            'Reviews Rating': 'mean',
            'Review Count': 'sum',
            'Sales Trend (90 days) (%)': 'mean'
        })
        
        # Normalize metrics
        normalized = pd.DataFrame()
        for column in brand_metrics.columns:
            normalized[column] = (brand_metrics[column] - brand_metrics[column].min()) / \
                               (brand_metrics[column].max() - brand_metrics[column].min())
        
        # Calculate strength index
        strength_index = (
            normalized['Monthly Sales'] * 0.3 +
            normalized['Reviews Rating'] * 0.25 +
            normalized['Review Count'] * 0.15 +
            normalized['Sales Trend (90 days) (%)'] * 0.2 +
            (1 - normalized['Price']) * 0.1  # Lower price is better
        )
        
        return strength_index.round(3)

def run_competitive_analysis(self):
    """Run comprehensive competitive analysis"""
    st.header("Competitive Analysis Dashboard")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Market Position", "Brand Performance", "Competitive Strength",
         "Price Analysis", "Growth Analysis", "Customer Perception"]
    )
    
    if analysis_type == "Market Position":
        self.analyze_market_position()
    elif analysis_type == "Brand Performance":
        self.analyze_brand_performance()
    elif analysis_type == "Competitive Strength":
        self.analyze_competitive_strength()
    elif analysis_type == "Price Analysis":
        self.analyze_price_competition()
    elif analysis_type == "Growth Analysis":
        self.analyze_growth_trends()
    elif analysis_type == "Customer Perception":
        self.analyze_customer_perception()


def analyze_market_position(self):
    """Analyze market positioning of brands"""
    st.subheader("Market Position Analysis")
    
    # Access the 'brand_performance' metric and ensure it's correctly formatted
    brand_performance = self.metrics['brand_performance'].reset_index()

    # Check if all required columns are present and consistent in length
    x_data = brand_performance[('Price', 'mean')]
    y_data = brand_performance['Monthly Sales']
    size_data = brand_performance['Monthly Revenue']
    color_data = brand_performance['Reviews Rating']
    
    if len(x_data) == len(y_data) == len(size_data) == len(color_data):
        fig_position = px.scatter(
            brand_performance,
            x=('Price', 'mean'),
            y='Monthly Sales',
            size='Monthly Revenue',
            color='Reviews Rating',
            text='Brand',
            title="Market Position Map"
        )
        st.plotly_chart(fig_position, use_container_width=True)
        self.plots['position_map'] = fig_position
    else:
        st.error("Data length mismatch: Ensure 'Price', 'Monthly Sales', 'Monthly Revenue', and 'Reviews Rating' columns have the same length.")

    # Display market position insights
    st.subheader("Market Position Insights")
    market_leader = self.metrics['market_share'].idxmax()
    premium_brand = self.metrics['brand_performance'][('Price', 'mean')].idxmax()
    
    st.write(f"• Market Leader: {market_leader} with {self.metrics['market_share'][market_leader]:.1f}% market share")
    st.write(f"• Premium Brand: {premium_brand} with average price ₹{self.metrics['brand_performance'].loc[premium_brand, ('Price', 'mean')]:,.2f}")


    def analyze_brand_performance(self):
        """Analyze detailed brand performance metrics"""
        st.subheader("Brand Performance Analysis")
        
        # Performance metrics table
        st.write("Brand Performance Metrics:")
        st.dataframe(self.metrics['brand_performance'])
        
        # Performance radar chart
        selected_brand = st.selectbox("Select Brand for Detailed Analysis", self.df['Brand'].unique())
        
        brand_data = self.metrics['brand_performance'].loc[selected_brand]
        
        # Create radar chart
        categories = ['Sales', 'Revenue', 'Rating', 'Growth', 'Price']
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                brand_data['Monthly Sales'],
                brand_data['Monthly Revenue'],
                brand_data['Reviews Rating'],
                brand_data['Sales Trend (90 days) (%)'],
                brand_data[('Price', 'mean')]
            ],
            theta=categories,
            fill='toself',
            name=selected_brand
        ))
        
        fig_radar.update_layout(title=f"{selected_brand} Performance Radar")
        st.plotly_chart(fig_radar, use_container_width=True)
        self.plots['brand_radar'] = fig_radar

    def analyze_competitive_strength(self):
        """Analyze competitive strength of brands"""
        st.subheader("Competitive Strength Analysis")
        
        # Strength index visualization
        fig_strength = px.bar(
            x=self.metrics['strength_index'].index,
            y=self.metrics['strength_index'].values,
            title="Competitive Strength Index"
        )
        st.plotly_chart(fig_strength, use_container_width=True)
        self.plots['strength_index'] = fig_strength
        
        # Detailed strength analysis
        st.write("Strength Index Components:")
        
        strength_components = pd.DataFrame({
            'Market Share': self.metrics['market_share'],
            'Rating Score': self.df.groupby('Brand')['Reviews Rating'].mean() * 20,
            'Growth Score': self.df.groupby('Brand')['Sales Trend (90 days) (%)'].mean(),
            'Overall Strength': self.metrics['strength_index']
        }).round(2)
        
        st.dataframe(strength_components)

    def analyze_price_competition(self):
        """Analyze price competition and positioning"""
        st.subheader("Price Competition Analysis")
        
        # Price distribution by brand
        fig_price = px.box(
            self.df,
            x='Brand',
            y='Price',
            title="Price Distribution by Brand"
        )
        st.plotly_chart(fig_price, use_container_width=True)
        self.plots['price_distribution'] = fig_price
        
        # Price segment analysis
        price_segments = pd.qcut(self.df['Price'], q=5, labels=[
            'Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury'
        ])
        
        segment_share = pd.crosstab(
            self.df['Brand'],
            price_segments,
            values=self.df['Monthly Sales'],
            aggfunc='sum',
            normalize='columns'
        ) * 100
        
        st.write("Price Segment Market Share (%):")
        st.dataframe(segment_share.round(2))

    def analyze_growth_trends(self):
        """Analyze growth trends and momentum"""
        st.subheader("Growth Trend Analysis")
        
        # Growth comparison
        fig_growth = px.scatter(
            self.df,
            x='Sales Trend (90 days) (%)',
            y='Price Trend (90 days) (%)',
            color='Brand',
            size='Monthly Sales',
            title="Growth Trends Comparison"
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        self.plots['growth_trends'] = fig_growth
        
        # Growth metrics
        growth_metrics = self.df.groupby('Brand').agg({
            'Sales Trend (90 days) (%)': ['mean', 'std'],
            'Price Trend (90 days) (%)': ['mean', 'std']
        }).round(2)
        
        st.write("Growth Metrics:")
        st.dataframe(growth_metrics)

    def analyze_customer_perception(self):
        """Analyze customer perception and satisfaction"""
        st.subheader("Customer Perception Analysis")
        
        # Rating distribution
        fig_rating = px.violin(
            self.df,
            x='Brand',
            y='Reviews Rating',
            box=True,
            title="Rating Distribution by Brand"
        )
        st.plotly_chart(fig_rating, use_container_width=True)
        self.plots['rating_distribution'] = fig_rating
        
        # Customer satisfaction metrics
        satisfaction_metrics = self.df.groupby('Brand').agg({
            'Reviews Rating': ['mean', 'count'],
            'Review Count': 'sum',
            'Monthly Sales': 'sum'
        }).round(2)
        
        satisfaction_metrics['Sales per Review'] = (
            satisfaction_metrics['Monthly Sales'] / 
            satisfaction_metrics['Review Count']
        ).round(2)
        
        st.write("Customer Satisfaction Metrics:")
        st.dataframe(satisfaction_metrics)

def add_competitive_analysis_section():
    """Add competitive analysis section to the app"""
    if 'data' in st.session_state:
        analyzer = CompetitiveAnalyzer(st.session_state['data'])
        analyzer.run_competitive_analysis()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_competitive_analysis_section()

# Part 5: Price Optimization and Revenue Analysis

class PriceOptimizer:
    def __init__(self, df):
        self.df = df
        self.plots = {}
        self.price_metrics = self._calculate_price_metrics()

    def _calculate_price_metrics(self):
        """Calculate comprehensive price metrics"""
        metrics = {}
        
        # Basic price metrics
        metrics['basic'] = {
            'mean_price': self.df['Price'].mean(),
            'median_price': self.df['Price'].median(),
            'price_range': (self.df['Price'].min(), self.df['Price'].max()),
            'price_std': self.df['Price'].std()
        }
        
        # Price elasticity by brand
        metrics['elasticity'] = self._calculate_price_elasticity()
        
        # Price segments
        metrics['segments'] = pd.qcut(
            self.df['Price'],
            q=5,
            labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury']
        )
        
        return metrics

    def _calculate_price_elasticity(self):
        """Calculate price elasticity of demand"""
        elasticity = {}
        
        for brand in self.df['Brand'].unique():
            brand_data = self.df[self.df['Brand'] == brand]
            
            if len(brand_data) > 1:
                log_price = np.log(brand_data['Price'])
                log_sales = np.log(brand_data['Monthly Sales'])
                
                slope, _, _, _, _ = stats.linregress(log_price, log_sales)
                elasticity[brand] = -slope
        
        return elasticity

    def run_price_optimization(self):
        """Run comprehensive price optimization analysis"""
        st.header("Price Optimization Dashboard")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Price Elasticity", "Revenue Optimization", "Margin Analysis",
             "Price Sensitivity", "Competitive Pricing", "Price Segmentation"]
        )
        
        if analysis_type == "Price Elasticity":
            self.analyze_price_elasticity()
        elif analysis_type == "Revenue Optimization":
            self.analyze_revenue_optimization()
        elif analysis_type == "Margin Analysis":
            self.analyze_margins()
        elif analysis_type == "Price Sensitivity":
            self.analyze_price_sensitivity()
        elif analysis_type == "Competitive Pricing":
            self.analyze_competitive_pricing()
        elif analysis_type == "Price Segmentation":
            self.analyze_price_segmentation()

    def analyze_price_elasticity(self):
        """Analyze price elasticity of demand"""
        st.subheader("Price Elasticity Analysis")
        
        # Overall market elasticity
        log_price = np.log(self.df['Price'])
        log_sales = np.log(self.df['Monthly Sales'])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_sales)
        
        # Display overall elasticity
        st.metric(
            "Market Price Elasticity",
            f"{-slope:.2f}",
            help="Values > 1 indicate elastic demand, < 1 indicate inelastic demand"
        )
        
        # Brand-wise elasticity
        elasticity_df = pd.DataFrame.from_dict(
            self.price_metrics['elasticity'],
            orient='index',
            columns=['Elasticity']
        )
        
        fig_elasticity = px.bar(
            elasticity_df,
            y='Elasticity',
            title="Price Elasticity by Brand"
        )
        st.plotly_chart(fig_elasticity, use_container_width=True)
        self.plots['elasticity'] = fig_elasticity
        
        # Elasticity interpretation
        st.subheader("Elasticity Insights")
        for brand, elasticity in self.price_metrics['elasticity'].items():
            sensitivity = "highly sensitive" if elasticity > 1.5 else \
                         "moderately sensitive" if elasticity > 1 else "less sensitive"
            st.write(f"• {brand}: {sensitivity} to price changes (elasticity: {elasticity:.2f})")

    def analyze_revenue_optimization(self):
        """Analyze revenue optimization opportunities"""
        st.subheader("Revenue Optimization Analysis")
        
        # Current revenue metrics
        current_revenue = self.df.groupby('Brand').agg({
            'Monthly Revenue': 'sum',
            'Price': 'mean',
            'Monthly Sales': 'sum'
        })
        
        # Optimize revenue using elasticity
        optimized_prices = {}
        potential_revenue = {}
        
        for brand in self.df['Brand'].unique():
            if brand in self.price_metrics['elasticity']:
                elasticity = self.price_metrics['elasticity'][brand]
                current_price = current_revenue.loc[brand, 'Price']
                
                # Optimal price calculation using elasticity
                optimal_price = current_price * (1 / (1 + 1/elasticity))
                optimized_prices[brand] = optimal_price
                
                # Calculate potential revenue
                current_sales = current_revenue.loc[brand, 'Monthly Sales']
                potential_sales = current_sales * (optimal_price/current_price)**(-elasticity)
                potential_revenue[brand] = optimal_price * potential_sales
        
        # Display optimization results
        optimization_results = pd.DataFrame({
            'Current Price': current_revenue['Price'],
            'Optimal Price': optimized_prices,
            'Current Revenue': current_revenue['Monthly Revenue'],
            'Potential Revenue': potential_revenue
        })
        
        st.write("Price Optimization Results:")
        st.dataframe(optimization_results.round(2))
        
        # Visualization of potential gains
        fig_optimization = go.Figure()
        fig_optimization.add_trace(go.Bar(
            name='Current Revenue',
            x=optimization_results.index,
            y=optimization_results['Current Revenue']
        ))
        fig_optimization.add_trace(go.Bar(
            name='Potential Revenue',
            x=optimization_results.index,
            y=optimization_results['Potential Revenue']
        ))
        
        fig_optimization.update_layout(
            barmode='group',
            title="Current vs Potential Revenue"
        )
        st.plotly_chart(fig_optimization, use_container_width=True)
        self.plots['revenue_optimization'] = fig_optimization

    def analyze_margins(self):
        """Analyze profit margins and optimization"""
        st.subheader("Margin Analysis")
        
        # Margin simulator
        margin_percent = st.slider("Estimated Gross Margin (%)", 0, 100, 30)
        
        # Calculate margins
        self.df['Estimated_Cost'] = self.df['Price'] * (1 - margin_percent/100)
        self.df['Margin_Amount'] = self.df['Price'] - self.df['Estimated_Cost']
        self.df['Total_Margin'] = self.df['Margin_Amount'] * self.df['Monthly Sales']
        
        # Margin analysis by brand
        margin_analysis = self.df.groupby('Brand').agg({
            'Margin_Amount': 'mean',
            'Total_Margin': 'sum',
            'Price': 'mean',
            'Monthly Sales': 'sum'
        }).round(2)
        
        st.write("Margin Analysis by Brand:")
        st.dataframe(margin_analysis)
        
        # Margin visualization
        fig_margins = px.scatter(
            self.df,
            x='Price',
            y='Total_Margin',
            size='Monthly Sales',
            color='Brand',
            title="Price vs Total Margin"
        )
        st.plotly_chart(fig_margins, use_container_width=True)
        self.plots['margins'] = fig_margins

    def analyze_price_sensitivity(self):
        """Analyze price sensitivity patterns"""
        st.subheader("Price Sensitivity Analysis")
        
        # Create price bands
        price_bands = pd.qcut(self.df['Price'], q=10)
        sensitivity_analysis = self.df.groupby(price_bands).agg({
            'Monthly Sales': 'mean',
            'Reviews Rating': 'mean',
            'Sales Trend (90 days) (%)': 'mean'
        }).round(2)
        
        st.write("Price Band Analysis:")
        st.dataframe(sensitivity_analysis)
        
        # Price sensitivity visualization
        fig_sensitivity = go.Figure()
        
        fig_sensitivity.add_trace(go.Scatter(
            x=[str(x) for x in sensitivity_analysis.index],
            y=sensitivity_analysis['Monthly Sales'],
            name='Average Sales'
        ))
        
        fig_sensitivity.update_layout(
            title="Sales Response to Price Bands",
            xaxis_title="Price Bands",
            yaxis_title="Average Monthly Sales"
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        self.plots['sensitivity'] = fig_sensitivity

    def analyze_competitive_pricing(self):
        """Analyze competitive pricing strategies"""
        st.subheader("Competitive Pricing Analysis")
        
        # Price positioning map
        fig_positioning = px.scatter(
            self.df,
            x='Price',
            y='Reviews Rating',
            size='Monthly Sales',
            color='Brand',
            title="Price-Quality Positioning"
        )
        st.plotly_chart(fig_positioning, use_container_width=True)
        self.plots['positioning'] = fig_positioning
        
        # Competitive price analysis
        competitive_analysis = self.df.groupby('Brand').agg({
            'Price': ['mean', 'min', 'max', 'std'],
            'Monthly Sales': 'sum',
            'Reviews Rating': 'mean'
        }).round(2)
        
        st.write("Competitive Price Analysis:")
        st.dataframe(competitive_analysis)

    def analyze_price_segmentation(self):
        """Analyze price segmentation strategies"""
        st.subheader("Price Segmentation Analysis")
        
        # Segment performance
        segment_performance = self.df.groupby(self.price_metrics['segments']).agg({
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean',
            'Price': ['mean', 'min', 'max']
        }).round(2)
        
        st.write("Segment Performance Metrics:")
        st.dataframe(segment_performance)
        
        # Segment visualization
        fig_segments = px.sunburst(
            self.df,
            path=[self.price_metrics['segments'], 'Brand'],
            values='Monthly Revenue',
            title="Revenue Distribution by Price Segment"
        )
        st.plotly_chart(fig_segments, use_container_width=True)
        self.plots['segments'] = fig_segments

def add_price_optimization_section():
    """Add price optimization section to the app"""
    if 'data' in st.session_state:
        optimizer = PriceOptimizer(st.session_state['data'])
        optimizer.run_price_optimization()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_price_optimization_section()

# Part 6: Seller Performance Analysis

class SellerAnalyzer:
    def __init__(self, df):
        self.df = df
        self.plots = {}
        self.seller_metrics = self._calculate_seller_metrics()

    def _calculate_seller_metrics(self):
        """Calculate comprehensive seller performance metrics"""
        metrics = {}
        
        # Basic performance metrics
        metrics['performance'] = self.df.groupby('Seller').agg({
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean',
            'Price': ['mean', 'min', 'max'],
            'Review Count': 'sum',
            'Sales Trend (90 days) (%)': 'mean'
        })
        
        # Market share
        total_revenue = self.df['Monthly Revenue'].sum()
        metrics['market_share'] = (
            self.df.groupby('Seller')['Monthly Revenue'].sum() / total_revenue * 100
        ).round(2)
        
        # Efficiency metrics
        metrics['efficiency'] = self._calculate_efficiency_metrics()
        
        # Portfolio metrics
        metrics['portfolio'] = self._analyze_portfolio()
        
        return metrics

    def _calculate_efficiency_metrics(self):
        """Calculate seller efficiency metrics"""
        efficiency = pd.DataFrame()
        
        # Group by seller
        seller_data = self.df.groupby('Seller')
        
        # Revenue per product
        efficiency['Revenue_per_Product'] = seller_data['Monthly Revenue'].sum() / seller_data.size()
        
        # Sales to review ratio
        efficiency['Sales_per_Review'] = seller_data['Monthly Sales'].sum() / seller_data['Review Count'].sum()
        
        # Average product rating
        efficiency['Avg_Rating'] = seller_data['Reviews Rating'].mean()
        
        # Growth rate
        efficiency['Growth_Rate'] = seller_data['Sales Trend (90 days) (%)'].mean()
        
        return efficiency.round(2)

    def _analyze_portfolio(self):
        """Analyze seller product portfolio"""
        portfolio = pd.DataFrame()
        
        # Product count
        portfolio['Product_Count'] = self.df.groupby('Seller').size()
        
        # Brand diversity
        portfolio['Brand_Count'] = self.df.groupby('Seller')['Brand'].nunique()
        
        # Price range coverage
        portfolio['Price_Range'] = self.df.groupby('Seller').apply(
            lambda x: x['Price'].max() - x['Price'].min()
        )
        
        # Category coverage
        portfolio['Category_Count'] = self.df.groupby('Seller')['Category'].nunique()
        
        return portfolio.round(2)

    def run_seller_analysis(self):
        """Run comprehensive seller performance analysis"""
        st.header("Seller Performance Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Performance Overview", "Portfolio Analysis", "Efficiency Metrics",
             "Growth Analysis", "Competitive Comparison", "Recommendations"]
        )
        
        if analysis_type == "Performance Overview":
            self.show_performance_overview()
        elif analysis_type == "Portfolio Analysis":
            self.analyze_seller_portfolio()
        elif analysis_type == "Efficiency Metrics":
            self.analyze_efficiency()
        elif analysis_type == "Growth Analysis":
            self.analyze_growth_trends()
        elif analysis_type == "Competitive Comparison":
            self.compare_sellers()
        elif analysis_type == "Recommendations":
            self.generate_recommendations()

    def show_performance_overview(self):
        """Show overall seller performance metrics"""
        st.subheader("Seller Performance Overview")
        
        # Select seller for detailed analysis
        selected_seller = st.selectbox("Select Seller", self.df['Seller'].unique())
        
        # Key metrics for selected seller
        seller_data = self.seller_metrics['performance'].loc[selected_seller]
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Monthly Revenue",
                f"₹{seller_data['Monthly Revenue']:,.0f}",
                f"{seller_data['Sales Trend (90 days) (%)']:.1f}% Growth"
            )
        
        with col2:
            st.metric(
                "Monthly Sales",
                f"{seller_data['Monthly Sales']:,.0f}",
                f"Avg Price: ₹{seller_data[('Price', 'mean')]:,.2f}"
            )
        
        with col3:
            st.metric(
                "Average Rating",
                f"{seller_data['Reviews Rating']:.2f}",
                f"{seller_data['Review Count']} Reviews"
            )
        
        with col4:
            market_share = self.seller_metrics['market_share'][selected_seller]
            st.metric(
                "Market Share",
                f"{market_share:.1f}%"
            )
        
        # Performance trends
        self._show_performance_trends(selected_seller)

    def _show_performance_trends(self, seller):
        """Show performance trends for selected seller"""
        seller_data = self.df[self.df['Seller'] == seller]
        
        # Sales trend
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Scatter(
            x=seller_data['Date'],
            y=seller_data['Monthly Sales'],
            name='Monthly Sales'
        ))
        
        fig_trends.update_layout(title=f"Sales Trend - {seller}")
        st.plotly_chart(fig_trends, use_container_width=True)
        self.plots['sales_trend'] = fig_trends

    def analyze_seller_portfolio(self):
        """Analyze seller product portfolio"""
        st.subheader("Portfolio Analysis")
        
        # Portfolio metrics
        st.write("Portfolio Metrics by Seller:")
        st.dataframe(self.seller_metrics['portfolio'])
        
        # Portfolio visualization
        selected_seller = st.selectbox("Select Seller for Portfolio Analysis", 
                                     self.df['Seller'].unique())
        
        # Brand distribution
        seller_portfolio = self.df[self.df['Seller'] == selected_seller]
        
        fig_portfolio = px.treemap(
            seller_portfolio,
            path=['Brand', 'Category'],
            values='Monthly Revenue',
            title=f"Portfolio Distribution - {selected_seller}"
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        self.plots['portfolio'] = fig_portfolio

    def analyze_efficiency(self):
        """Analyze seller efficiency metrics"""
        st.subheader("Efficiency Analysis")
        
        # Efficiency metrics
        st.write("Efficiency Metrics:")
        st.dataframe(self.seller_metrics['efficiency'])
        
        # Efficiency visualization
        fig_efficiency = px.scatter(
            self.seller_metrics['efficiency'].reset_index(),
            x='Revenue_per_Product',
            y='Sales_per_Review',
            size='Avg_Rating',
            color='Growth_Rate',
            hover_data=['Seller'],
            title="Seller Efficiency Matrix"
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
        self.plots['efficiency'] = fig_efficiency

    def analyze_growth_trends(self):
        """Analyze seller growth trends"""
        st.subheader("Growth Analysis")
        
        # Growth metrics
        growth_metrics = pd.DataFrame({
            'Sales_Growth': self.seller_metrics['performance']['Sales Trend (90 days) (%)'],
            'Revenue_Growth': self.df.groupby('Seller')['Monthly Revenue'].pct_change(),
            'Rating_Trend': self.df.groupby('Seller')['Reviews Rating'].pct_change()
        })
        
        st.write("Growth Metrics:")
        st.dataframe(growth_metrics.round(2))
        
        # Growth visualization
        fig_growth = px.bar(
            growth_metrics.reset_index(),
            x='Seller',
            y='Sales_Growth',
            color='Revenue_Growth',
            title="Seller Growth Analysis"
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        self.plots['growth'] = fig_growth

    def compare_sellers(self):
        """Compare sellers performance"""
        st.subheader("Seller Comparison")
        
        # Select sellers to compare
        sellers_to_compare = st.multiselect(
            "Select Sellers to Compare",
            self.df['Seller'].unique(),
            default=list(self.df['Seller'].unique())[:2]
        )
        
        if len(sellers_to_compare) > 0:
            # Comparison metrics
            comparison_data = self.seller_metrics['performance'].loc[sellers_to_compare]
            
            # Radar chart comparison
            fig_comparison = go.Figure()
            
            for seller in sellers_to_compare:
                fig_comparison.add_trace(go.Scatterpolar(
                    r=[
                        comparison_data.loc[seller, 'Monthly Sales'],
                        comparison_data.loc[seller, 'Monthly Revenue'],
                        comparison_data.loc[seller, 'Reviews Rating'],
                        comparison_data.loc[seller, ('Price', 'mean')],
                        comparison_data.loc[seller, 'Sales Trend (90 days) (%)']
                    ],
                    theta=['Sales', 'Revenue', 'Rating', 'Price', 'Growth'],
                    name=seller
                ))
            
            fig_comparison.update_layout(title="Seller Comparison Radar")
            st.plotly_chart(fig_comparison, use_container_width=True)
            self.plots['comparison'] = fig_comparison

    def generate_recommendations(self):
        """Generate recommendations for sellers"""
        st.subheader("Seller Recommendations")
        
        selected_seller = st.selectbox(
            "Select Seller for Recommendations",
            self.df['Seller'].unique()
        )
        
        recommendations = self._get_seller_recommendations(selected_seller)
        
        for category, items in recommendations.items():
            with st.expander(category):
                for item in items:
                    st.write(f"• {item}")

    def _get_seller_recommendations(self, seller):
        """Generate specific recommendations for a seller"""
        seller_data = self.df[self.df['Seller'] == seller]
        metrics = self.seller_metrics
        
        recommendations = {
            "Portfolio Optimization": [],
            "Price Optimization": [],
            "Growth Opportunities": [],
            "Performance Improvement": []
        }
        
        # Portfolio recommendations
        portfolio_size = metrics['portfolio'].loc[seller, 'Product_Count']
        avg_portfolio = metrics['portfolio']['Product_Count'].mean()
        
        if portfolio_size < avg_portfolio:
            recommendations["Portfolio Optimization"].append(
                f"Consider expanding product portfolio (current: {portfolio_size:.0f}, average: {avg_portfolio:.0f})"
            )
        
        # Price recommendations
        price_efficiency = metrics['efficiency'].loc[seller, 'Revenue_per_Product']
        avg_efficiency = metrics['efficiency']['Revenue_per_Product'].mean()
        
        if price_efficiency < avg_efficiency:
            recommendations["Price Optimization"].append(
                "Optimize pricing strategy to improve revenue per product"
            )
        
        # Growth recommendations
        growth_rate = metrics['efficiency'].loc[seller, 'Growth_Rate']
        if growth_rate < metrics['efficiency']['Growth_Rate'].mean():
            recommendations["Growth Opportunities"].append(
                "Focus on growth strategies to improve market position"
            )
        
        # Performance recommendations
        rating = metrics['performance'].loc[seller, 'Reviews Rating']
        if rating < metrics['performance']['Reviews Rating'].mean():
            recommendations["Performance Improvement"].append(
                "Implement customer satisfaction initiatives to improve ratings"
            )
        
        return recommendations

def add_seller_analysis_section():
    """Add seller analysis section to the app"""
    if 'data' in st.session_state:
        analyzer = SellerAnalyzer(st.session_state['data'])
        analyzer.run_seller_analysis()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_seller_analysis_section()

# Part 7: Report Generation and Export

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import io
import base64
import json
import xlsxwriter
from datetime import datetime

class ReportGenerator:
    def __init__(self, df, analysis_results):
        self.df = df
        self.analysis_results = analysis_results
        self.styles = getSampleStyleSheet()
        self.plots = {}

    def generate_report_ui(self):
        """Create report generation interface"""
        st.header("Report Generation")

        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Full Analysis Report", "Custom Report", 
             "Seller Performance Report", "Market Analysis Report"]
        )

        # Report customization options
        with st.expander("Report Customization Options"):
            include_sections = st.multiselect(
                "Select Sections to Include",
                ["Market Overview", "Price Analysis", "Competitor Analysis",
                 "Seller Performance", "Growth Trends", "Recommendations"],
                default=["Market Overview", "Price Analysis"]
            )

            chart_style = st.selectbox(
                "Chart Style",
                ["Modern", "Classic", "Minimal"]
            )

            include_raw_data = st.checkbox("Include Raw Data Tables")
            include_methodology = st.checkbox("Include Methodology Section")

        if st.button("Generate Report"):
            report_buffer = self._generate_report(
                report_type,
                include_sections,
                chart_style,
                include_raw_data,
                include_methodology
            )
            
            # Offer download options
            self._create_download_buttons(report_buffer, report_type)

    def _generate_report(self, report_type, sections, chart_style, include_raw_data, include_methodology):
        """Generate the selected report type"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Add report header
        story.extend(self._create_header(report_type))

        # Add executive summary
        story.extend(self._create_executive_summary())

        # Add selected sections
        for section in sections:
            if section == "Market Overview":
                story.extend(self._create_market_overview_section())
            elif section == "Price Analysis":
                story.extend(self._create_price_analysis_section())
            elif section == "Competitor Analysis":
                story.extend(self._create_competitor_analysis_section())
            elif section == "Seller Performance":
                story.extend(self._create_seller_performance_section())
            elif section == "Growth Trends":
                story.extend(self._create_growth_trends_section())
            elif section == "Recommendations":
                story.extend(self._create_recommendations_section())

        # Add methodology if requested
        if include_methodology:
            story.extend(self._create_methodology_section())

        # Add raw data if requested
        if include_raw_data:
            story.extend(self._create_raw_data_section())

        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    def _create_header(self, report_type):
        """Create report header"""
        story = []
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )

        # Add title
        story.append(Paragraph(f"Baby Stroller Market Analysis", title_style))
        story.append(Paragraph(f"{report_type}", self.styles['Heading2']))
        story.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 20))

        return story

    def _create_executive_summary(self):
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['Heading1']))
        story.append(Spacer(1, 10))

        # Key metrics
        metrics = [
            ["Total Products", f"{len(self.df):,}"],
            ["Total Revenue", f"₹{self.df['Monthly Revenue'].sum():,.2f}"],
            ["Average Rating", f"{self.df['Reviews Rating'].mean():.2f}"],
            ["Growth Rate", f"{self.df['Sales Trend (90 days) (%)'].mean():.1f}%"]
        ]

        # Create table
        table = Table(metrics)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        return story

    def _create_market_overview_section(self):
        """Create market overview section"""
        story = []
        
        story.append(Paragraph("Market Overview", self.styles['Heading1']))
        story.append(Spacer(1, 10))

        # Market share chart
        market_share = self.df.groupby('Brand')['Monthly Revenue'].sum()
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 100
        pie.height = 100
        pie.data = market_share.values
        pie.labels = market_share.index
        drawing.add(pie)
        story.append(drawing)

        # Market insights
        insights = self._generate_market_insights()
        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles['Normal']))
            story.append(Spacer(1, 5))

        return story

    def _create_price_analysis_section(self):
        """Create price analysis section"""
        story = []
        
        story.append(Paragraph("Price Analysis", self.styles['Heading1']))
        story.append(Spacer(1, 10))

        # Price distribution chart
        price_stats = self.df.groupby('Brand')['Price'].agg(['mean', 'min', 'max'])
        
        # Create table
        table_data = [['Brand', 'Average Price', 'Min Price', 'Max Price']]
        for brand in price_stats.index:
            table_data.append([
                brand,
                f"₹{price_stats.loc[brand, 'mean']:,.2f}",
                f"₹{price_stats.loc[brand, 'min']:,.2f}",
                f"₹{price_stats.loc[brand, 'max']:,.2f}"
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        return story

    def _generate_market_insights(self):
        """Generate market insights"""
        insights = []
        
        # Market leader
        market_leader = self.df.groupby('Brand')['Monthly Revenue'].sum().idxmax()
        market_share = (self.df[self.df['Brand'] == market_leader]['Monthly Revenue'].sum() / 
                       self.df['Monthly Revenue'].sum() * 100)
        
        insights.append(f"Market Leader: {market_leader} with {market_share:.1f}% market share")
        
        # Price insights
        avg_price = self.df['Price'].mean()
        price_trend = self.df['Price Trend (90 days) (%)'].mean()
        
        insights.append(f"Average Market Price: ₹{avg_price:,.2f} with {price_trend:+.1f}% trend")
        
        # Growth insights
        growth_rate = self.df['Sales Trend (90 days) (%)'].mean()
        insights.append(f"Market Growth Rate: {growth_rate:+.1f}%")

        return insights

    def _create_download_buttons(self, report_buffer, report_type):
        """Create download buttons for different formats"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PDF Download
            st.download_button(
                label="Download PDF Report",
                data=report_buffer,
                file_name=f"stroller_market_analysis_{report_type.lower()}.pdf",
                mime="application/pdf"
            )
        
        with col2:
            # Excel Export
            excel_buffer = self._generate_excel_report(report_type)
            st.download_button(
                label="Download Excel Report",
                data=excel_buffer,
                file_name=f"stroller_market_analysis_{report_type.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # JSON Export
            json_data = self._generate_json_report(report_type)
            st.download_button(
                label="Download JSON Data",
                data=json_data,
                file_name=f"stroller_market_analysis_{report_type.lower()}.json",
                mime="application/json"
            )

    def _generate_excel_report(self, report_type):
        """Generate Excel report"""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write main data
            self.df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Write summary metrics
            summary_metrics = pd.DataFrame({
                'Metric': ['Total Products', 'Total Revenue', 'Average Rating', 'Growth Rate'],
                'Value': [
                    len(self.df),
                    self.df['Monthly Revenue'].sum(),
                    self.df['Reviews Rating'].mean(),
                    self.df['Sales Trend (90 days) (%)'].mean()
                ]
            })
            summary_metrics.to_excel(writer, sheet_name='Summary', index=False)
            
            # Write analysis results
            if self.analysis_results:
                pd.DataFrame(self.analysis_results).to_excel(
                    writer,
                    sheet_name='Analysis Results',
                    index=False
                )
        
        buffer.seek(0)
        return buffer

    def _generate_json_report(self, report_type):
        """Generate JSON report"""
        report_data = {
            'report_type': report_type,
            'generation_date': datetime.now().isoformat(),
            'summary_metrics': {
                'total_products': len(self.df),
                'total_revenue': float(self.df['Monthly Revenue'].sum()),
                'average_rating': float(self.df['Reviews Rating'].mean()),
                'growth_rate': float(self.df['Sales Trend (90 days) (%)'].mean())
            },
            'analysis_results': self.analysis_results if self.analysis_results else {}
        }
        
        return json.dumps(report_data, indent=2)

def add_report_generation_section():
    """Add report generation section to the app"""
    if 'data' in st.session_state and 'analysis_results' in st.session_state:
        generator = ReportGenerator(
            st.session_state['data'],
            st.session_state['analysis_results']
        )
        generator.generate_report_ui()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_report_generation_section()

# Part 8: Decision Support System

class DecisionSupportSystem:
    def __init__(self, df):
        self.df = df
        self.metrics = self._calculate_decision_metrics()
        self.plots = {}

    def _calculate_decision_metrics(self):
        """Calculate comprehensive metrics for decision making"""
        metrics = {
            'market': self._analyze_market_conditions(),
            'competition': self._analyze_competitive_landscape(),
            'opportunities': self._identify_opportunities(),
            'risks': self._assess_risks()
        }
        return metrics

    def _analyze_market_conditions(self):
        """Analyze current market conditions"""
        return {
            'market_size': self.df['Monthly Revenue'].sum(),
            'growth_rate': self.df['Sales Trend (90 days) (%)'].mean(),
            'market_concentration': self._calculate_hhi(),
            'price_trend': self.df['Price Trend (90 days) (%)'].mean(),
            'avg_rating': self.df['Reviews Rating'].mean(),
            'market_segments': self._analyze_segments()
        }

    def _calculate_hhi(self):
        """Calculate Herfindahl-Hirschman Index"""
        market_shares = self.df.groupby('Brand')['Monthly Revenue'].sum()
        total_revenue = market_shares.sum()
        market_shares = (market_shares / total_revenue) * 100
        return (market_shares ** 2).sum()

    def _analyze_segments(self):
        """Analyze market segments"""
        self.df['Price_Segment'] = pd.qcut(
            self.df['Price'],
            q=5,
            labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury']
        )
        return self.df.groupby('Price_Segment').agg({
            'Monthly Sales': 'sum',
            'Monthly Revenue': 'sum',
            'Reviews Rating': 'mean'
        }).to_dict()

    def _analyze_competitive_landscape(self):
        """Analyze competitive landscape"""
        return {
            'brand_strength': self._calculate_brand_strength(),
            'price_positioning': self._analyze_price_positioning(),
            'market_leaders': self._identify_market_leaders(),
            'competitive_dynamics': self._analyze_competitive_dynamics()
        }

    def _calculate_brand_strength(self):
        """Calculate brand strength indices"""
        brand_metrics = self.df.groupby('Brand').agg({
            'Monthly Sales': 'sum',
            'Reviews Rating': 'mean',
            'Review Count': 'sum',
            'Sales Trend (90 days) (%)': 'mean'
        })
        
        # Normalize metrics
        normalized = (brand_metrics - brand_metrics.min()) / (brand_metrics.max() - brand_metrics.min())
        
        # Calculate strength index
        strength_index = (
            normalized['Monthly Sales'] * 0.4 +
            normalized['Reviews Rating'] * 0.3 +
            normalized['Review Count'] * 0.2 +
            normalized['Sales Trend (90 days) (%)'] * 0.1
        )
        
        return strength_index.to_dict()

    def run_decision_support(self):
        """Run decision support analysis"""
        st.header("Decision Support System")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Strategic Recommendations", "Risk Assessment", "Opportunity Analysis",
             "Action Planning", "Market Entry Analysis", "Portfolio Optimization"]
        )
        
        if analysis_type == "Strategic Recommendations":
            self.show_strategic_recommendations()
        elif analysis_type == "Risk Assessment":
            self.show_risk_assessment()
        elif analysis_type == "Opportunity Analysis":
            self.show_opportunity_analysis()
        elif analysis_type == "Action Planning":
            self.show_action_planning()
        elif analysis_type == "Market Entry Analysis":
            self.show_market_entry_analysis()
        elif analysis_type == "Portfolio Optimization":
            self.show_portfolio_optimization()

    def show_strategic_recommendations(self):
        """Generate and display strategic recommendations"""
        st.subheader("Strategic Recommendations")
        
        # Strategy context selection
        business_objective = st.selectbox(
            "Select Primary Business Objective",
            ["Market Share Growth", "Profit Maximization", "Brand Building",
             "Market Entry", "Risk Mitigation"]
        )
        
        # Additional context
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Very Low", "Low", "Moderate", "High", "Very High"]
        )
        
        timeframe = st.selectbox(
            "Implementation Timeframe",
            ["Short-term (0-6 months)", "Medium-term (6-12 months)", "Long-term (12+ months)"]
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            business_objective,
            risk_tolerance,
            timeframe
        )
        
        # Display recommendations
        for category, items in recommendations.items():
            with st.expander(category):
                for item in items:
                    st.write(f"• {item}")

    def _generate_recommendations(self, objective, risk_tolerance, timeframe):
        """Generate specific recommendations based on context"""
        recommendations = {
            "Strategic Initiatives": [],
            "Tactical Actions": [],
            "Risk Mitigation": [],
            "Growth Opportunities": []
        }
        
        market_conditions = self.metrics['market']
        
        # Strategic Initiatives
        if objective == "Market Share Growth":
            if market_conditions['growth_rate'] > 0:
                recommendations["Strategic Initiatives"].append(
                    "Expand product portfolio in high-growth segments"
                )
            else:
                recommendations["Strategic Initiatives"].append(
                    "Focus on market penetration in existing segments"
                )
        
        # Tactical Actions based on risk tolerance
        if risk_tolerance in ["High", "Very High"]:
            recommendations["Tactical Actions"].append(
                "Consider aggressive pricing strategy for market penetration"
            )
        else:
            recommendations["Tactical Actions"].append(
                "Implement gradual price optimization strategy"
            )
        
        return recommendations

    def show_risk_assessment(self):
        """Show risk assessment analysis"""
        st.subheader("Risk Assessment")
        
        # Risk matrix
        risks = self._identify_risks()
        
        # Create risk matrix visualization
        fig_risks = go.Figure()
        
        for risk in risks:
            fig_risks.add_trace(go.Scatter(
                x=[risk['probability']],
                y=[risk['impact']],
                mode='markers+text',
                name=risk['name'],
                text=[risk['name']],
                marker=dict(
                    size=risk['severity'] * 20,
                    color=risk['color']
                )
            ))
        
        fig_risks.update_layout(
            title="Risk Assessment Matrix",
            xaxis_title="Probability",
            yaxis_title="Impact"
        )
        
        st.plotly_chart(fig_risks, use_container_width=True)
        
        # Risk details
        for risk in risks:
            with st.expander(f"Risk: {risk['name']}"):
                st.write(f"Severity: {risk['severity']}")
                st.write(f"Mitigation Strategy: {risk['mitigation']}")

    def _identify_risks(self):
        """Identify and assess risks"""
        risks = []
        
        # Market concentration risk
        hhi = self.metrics['market']['market_concentration']
        risks.append({
            'name': 'Market Concentration Risk',
            'probability': 0.7 if hhi > 2500 else 0.4,
            'impact': 0.8 if hhi > 2500 else 0.5,
            'severity': 0.75 if hhi > 2500 else 0.45,
            'color': 'red' if hhi > 2500 else 'yellow',
            'mitigation': 'Diversify product portfolio and market segments'
        })
        
        # Add more risks based on metrics
        return risks

    def show_opportunity_analysis(self):
        """Show opportunity analysis"""
        st.subheader("Opportunity Analysis")
        
        # Opportunity score calculation
        opportunities = self._identify_opportunities()
        
        # Create opportunity visualization
        fig_opportunities = px.scatter(
            pd.DataFrame(opportunities),
            x='potential',
            y='feasibility',
            size='market_size',
            color='category',
            hover_data=['description'],
            title="Opportunity Matrix"
        )
        
        st.plotly_chart(fig_opportunities, use_container_width=True)
        
        # Opportunity details
        for opp in opportunities:
            with st.expander(f"Opportunity: {opp['name']}"):
                st.write(f"Category: {opp['category']}")
                st.write(f"Description: {opp['description']}")
                st.write(f"Potential: {opp['potential']:.2f}")
                st.write(f"Feasibility: {opp['feasibility']:.2f}")

    def show_action_planning(self):
        """Show action planning tools"""
        st.subheader("Action Planning")
        
        # Timeline selection
        timeline = st.selectbox(
            "Select Planning Timeline",
            ["30 Days", "90 Days", "6 Months", "1 Year"]
        )
        
        # Generate action plan
        action_plan = self._generate_action_plan(timeline)
        
        # Display action plan
        for phase, actions in action_plan.items():
            st.write(f"### {phase}")
            for action in actions:
                st.write(
                    f"• **{action['name']}**\n"
                    f"  Priority: {action['priority']}\n"
                    f"  Timeline: {action['timeline']}\n"
                    f"  Resources: {action['resources']}"
                )

    def show_market_entry_analysis(self):
        """Show market entry analysis"""
        st.subheader("Market Entry Analysis")
        
        # Segment selection
        selected_segment = st.selectbox(
            "Select Target Segment",
            ["Budget", "Economy", "Mid-Range", "Premium", "Luxury"]
        )
        
        # Entry strategy selection
        entry_strategy = st.selectbox(
            "Select Entry Strategy",
            ["New Brand Launch", "Partnership", "Acquisition", "Product Line Extension"]
        )
        
        # Generate entry analysis
        entry_analysis = self._analyze_market_entry(selected_segment, entry_strategy)
        
        # Display analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Market Attractiveness")
            for factor, score in entry_analysis['attractiveness'].items():
                st.metric(factor, f"{score:.2f}/5.0")
        
        with col2:
            st.write("### Entry Barriers")
            for barrier, impact in entry_analysis['barriers'].items():
                st.metric(barrier, f"{impact:.2f}/5.0")

    def show_portfolio_optimization(self):
        """Show portfolio optimization analysis"""
        st.subheader("Portfolio Optimization")
        
        # Current portfolio analysis
        portfolio_metrics = self._analyze_portfolio_metrics()
        
        # Display portfolio metrics
        st.write("### Current Portfolio Performance")
        st.dataframe(portfolio_metrics)
        
        # Optimization recommendations
        recommendations = self._generate_portfolio_recommendations(portfolio_metrics)
        
        st.write("### Optimization Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")

def add_decision_support_section():
    """Add decision support section to the app"""
    if 'data' in st.session_state:
        dss = DecisionSupportSystem(st.session_state['data'])
        dss.run_decision_support()

if __name__ == "__main__":
    if 'data' in st.session_state:
        add_decision_support_section()
