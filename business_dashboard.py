import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Timeline Events Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Load data functions
@st.cache_data
def load_parquet_file(file):
    """Load and preprocess the uploaded parquet file"""
    df = pd.read_parquet(file)
    
    # Add derived columns
    df['lifespan'] = df['end_year'] - df['start_year']
    df['lifespan'] = df['lifespan'].apply(lambda x: x if x > 0 else None)
    
    # Create a display year column (use start_year primarily)
    df['display_year'] = df['start_year'].fillna(df['end_year'])
    
    return df

@st.cache_data
def load_csv_file(file):
    """Load and preprocess the uploaded CSV file"""
    df = pd.read_csv(file)
    
    # Convert year columns to numeric if they aren't already
    for col in ['start_year', 'end_year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add derived columns
    df['lifespan'] = df['end_year'] - df['start_year']
    df['lifespan'] = df['lifespan'].apply(lambda x: x if x > 0 else None)
    
    # Create a display year column (use start_year primarily)
    df['display_year'] = df['start_year'].fillna(df['end_year'])
    
    return df

# Calculate KPIs
def calculate_kpis(df, filtered_df):
    """Calculate key performance indicators"""
    total_labels = df['label_name'].nunique()
    
    # Active labels: those without end_year or end_year in the future
    current_year = datetime.now().year
    active_labels = df[
        (df['end_year'].isna()) | (df['end_year'] >= current_year)
    ]['label_name'].nunique()
    
    # Average lifespan
    lifespans = df[df['lifespan'].notna()]['lifespan']
    avg_lifespan = lifespans.mean() if len(lifespans) > 0 else 0
    
    return {
        'total_labels': total_labels,
        'active_labels': active_labels,
        'avg_lifespan': avg_lifespan,
        'total_events': len(filtered_df)
    }

# Create line chart for timeline trends
def create_timeline_trends(df, selected_types, year_range):
    """Create line chart showing event trends over time"""
    
    # Filter data
    filtered = df[
        (df['event_type'].isin(selected_types)) &
        (df['display_year'] >= year_range[0]) &
        (df['display_year'] <= year_range[1])
    ]
    
    if len(filtered) == 0:
        return None, filtered
    
    # Aggregate by year and event type
    yearly_counts = filtered.groupby(['display_year', 'event_type']).size().reset_index(name='count')
    
    # Define color map
    color_map = {
        'Founded': '#2E7D32',
        'Merged': '#F57C00', 
        'Discontinued': '#D32F2F',
        'Rename': '#1976D2'
    }
    # Add colors for any other event types
    unique_types = df['event_type'].unique()
    default_colors = ['#7B1FA2', '#00796B', '#5D4037', '#616161']
    for i, event_type in enumerate(unique_types):
        if event_type not in color_map:
            color_map[event_type] = default_colors[i % len(default_colors)]
    
    # Create line chart
    fig = px.line(
        yearly_counts,
        x='display_year',
        y='count',
        color='event_type',
        color_discrete_map=color_map,
        title='Timeline Events Trends Over Time',
        labels={'display_year': 'Year', 'count': 'Number of Events', 'event_type': 'Event Type'},
        markers=True
    )
    
    # Update layout for better readability
    fig.update_layout(
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            title="Event Type",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add markers and improve line style
    fig.update_traces(
        mode='lines+markers',
        marker=dict(size=6),
        line=dict(width=2)
    )
    
    return fig, filtered

# Create improved heatmap
def create_heatmap(df, selected_types, year_range, grouping='year'):
    """Create improved heatmap with better readability"""
    
    # Filter data
    filtered = df[
        (df['event_type'].isin(selected_types)) &
        (df['display_year'] >= year_range[0]) &
        (df['display_year'] <= year_range[1])
    ]
    
    if len(filtered) == 0:
        return None
    
    # Group years based on selected grouping
    if grouping == '5 years':
        filtered['year_group'] = (filtered['display_year'] // 5) * 5
        group_col = 'year_group'
        x_label = 'Year (5-year groups)'
    elif grouping == '10 years':
        filtered['year_group'] = (filtered['display_year'] // 10) * 10
        group_col = 'year_group'
        x_label = 'Year (10-year groups)'
    else:
        group_col = 'display_year'
        x_label = 'Year'
    
    # Aggregate data
    heatmap_data = filtered.groupby([group_col, 'event_type']).size().reset_index(name='count')
    
    # Pivot for heatmap
    pivot_data = heatmap_data.pivot(index='event_type', columns=group_col, values='count').fillna(0)
    
    # Create heatmap with improved color scale
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{int(year)}" for year in pivot_data.columns],
        y=pivot_data.index,
        colorscale='Blues',
        text=pivot_data.values.astype(int),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Event Count"),
        hoverongaps=False,
        hovertemplate='Year: %{x}<br>Event Type: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Event Distribution Heatmap',
        xaxis_title=x_label,
        yaxis_title='Event Type',
        height=400,
        xaxis={'type': 'category'},
        yaxis={'type': 'category'}
    )
    
    return fig

# Create top companies table
def get_top_companies(df, n=10):
    """Get top N most active companies"""
    
    company_stats = df.groupby('label_name').agg({
        'event_type': 'count',
        'start_year': 'min',
        'display_year': 'max'
    }).reset_index()
    
    company_stats.columns = ['Company Name', 'Event Count', 'Founded Year', 'Latest Event Year']
    company_stats = company_stats.sort_values('Event Count', ascending=False).head(n)
    
    # Format years
    company_stats['Founded Year'] = company_stats['Founded Year'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    company_stats['Latest Event Year'] = company_stats['Latest Event Year'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    
    return company_stats

# Main application
def main():
    st.title("Timeline Events Dashboard")
    st.markdown("Interactive visualization of company timeline events")
    
    # File upload section
    st.markdown("### Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a file (Parquet or CSV format)",
        type=['parquet', 'csv'],
        help="Upload your timeline events data file. The file should contain columns: label_name, source, event_type, start_year, end_year, etc."
    )
    
    # Sample data description
    with st.expander("Expected Data Format"):
        st.markdown("""
        **Required columns:**
        - `label_name`: Company/entity name
        - `event_type`: Type of event (e.g., Founded, Merged, Discontinued)
        - `start_year`: Starting year of the event
        - `end_year`: Ending year of the event (if applicable)
        - `source`: Data source
        
        **Optional columns:**
        - `trigger`, `new_name`, `raw_time`, `start_month`, `end_month`
        - `granularity`, `confidence`, `sentence`, `profile`
        """)
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.parquet'):
                df = load_parquet_file(uploaded_file)
            else:
                df = load_csv_file(uploaded_file)
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Success message
            st.success(f"Successfully loaded {len(df):,} records from {df['label_name'].nunique():,} companies")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please ensure your file has the required columns and is in the correct format.")
            return
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.info("Please upload a data file to begin.")
        
        # Show sample data structure
        st.markdown("### Sample Data Structure")
        sample_data = pd.DataFrame({
            'label_name': ['Company A', 'Company A', 'Company B'],
            'event_type': ['Founded', 'Merged', 'Founded'],
            'start_year': [2010, 2015, 2012],
            'end_year': [None, None, 2018],
            'source': ['Source1', 'Source2', 'Source1'],
            'confidence': [0.95, 0.87, 0.92]
        })
        st.dataframe(sample_data)
        return
    
    df = st.session_state.df
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Year range slider - default to 1950-2025 if data allows
    min_year = df['display_year'].min()
    max_year = df['display_year'].max()
    
    # Set sensible defaults
    default_min = max(1950, int(min_year))
    default_max = min(2025, int(max_year))
    
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(default_min, default_max),
        step=1
    )
    
    # Event type filter
    event_types = df['event_type'].unique().tolist()
    selected_types = st.sidebar.multiselect(
        "Select Event Types",
        options=event_types,
        default=event_types
    )
    
    # Filter dataframe
    filtered_df = df[
        (df['event_type'].isin(selected_types)) &
        (df['display_year'] >= year_range[0]) &
        (df['display_year'] <= year_range[1])
    ]
    
    # Calculate KPIs
    kpis = calculate_kpis(df, filtered_df)
    
    # Display KPIs
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Companies", f"{kpis['total_labels']:,}")
    
    with col2:
        st.metric("Active Companies", f"{kpis['active_labels']:,}")
    
    with col3:
        st.metric("Average Lifespan", f"{kpis['avg_lifespan']:.1f} years")
    
    with col4:
        st.metric("Total Events (Filtered)", f"{kpis['total_events']:,}")
    
    # Main visualizations
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline Trends", "Heatmap", "Top Companies", "Data Explorer"])
    
    with tab1:
        # Line chart for trends
        if selected_types:
            st.markdown("#### Event Trends Over Time")
            
            # Add view options
            col1, col2 = st.columns([3, 1])
            with col2:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Line Chart", "Area Chart", "Stacked Area"],
                    key="chart_type_select"
                )
            
            fig_trends, trends_data = create_timeline_trends(df, selected_types, year_range)
            
            if fig_trends is not None:
                # Modify chart based on selection
                if chart_type == "Area Chart":
                    yearly_counts = trends_data.groupby(['display_year', 'event_type']).size().reset_index(name='count')
                    fig_trends = px.area(
                        yearly_counts,
                        x='display_year',
                        y='count',
                        color='event_type',
                        title='Timeline Events Trends Over Time',
                        labels={'display_year': 'Year', 'count': 'Number of Events'}
                    )
                elif chart_type == "Stacked Area":
                    yearly_counts = trends_data.groupby(['display_year', 'event_type']).size().reset_index(name='count')
                    pivot_data = yearly_counts.pivot(index='display_year', columns='event_type', values='count').fillna(0)
                    
                    fig_trends = go.Figure()
                    for col in pivot_data.columns:
                        fig_trends.add_trace(go.Scatter(
                            x=pivot_data.index,
                            y=pivot_data[col],
                            mode='lines',
                            stackgroup='one',
                            name=col,
                            fill='tonexty'
                        ))
                    
                    fig_trends.update_layout(
                        title='Timeline Events Trends Over Time (Stacked)',
                        xaxis_title='Year',
                        yaxis_title='Number of Events',
                        height=500,
                        hovermode='x unified'
                    )
                
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### Trend Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_events = len(trends_data)
                    st.info(f"Total Events in Period: {total_events:,}")
                
                with col2:
                    if len(trends_data) > 0:
                        peak_year = trends_data.groupby('display_year').size().idxmax()
                        peak_count = trends_data.groupby('display_year').size().max()
                        st.info(f"Peak Year: {peak_year:.0f} ({peak_count} events)")
                
                with col3:
                    if len(trends_data) > 0:
                        avg_per_year = len(trends_data) / (year_range[1] - year_range[0] + 1)
                        st.info(f"Average Events/Year: {avg_per_year:.1f}")
            else:
                st.warning("No data available for the selected filters")
        else:
            st.warning("Please select at least one event type")
    
    with tab2:
        # Improved Heatmap
        if selected_types:
            st.markdown("#### Event Distribution Heatmap")
            
            # Add grouping option
            col1, col2 = st.columns([3, 1])
            with col2:
                grouping = st.selectbox(
                    "Time Grouping",
                    ["year", "5 years", "10 years"],
                    key="heatmap_grouping"
                )
            
            fig_heatmap = create_heatmap(df, selected_types, year_range, grouping)
            
            if fig_heatmap is not None:
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Additional insights
                st.markdown("#### Distribution Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(filtered_df) > 0:
                        most_common_type = filtered_df['event_type'].value_counts().index[0]
                        type_count = filtered_df['event_type'].value_counts().values[0]
                        type_pct = (type_count / len(filtered_df)) * 100
                        st.info(f"Most Common Event: {most_common_type} ({type_count} events, {type_pct:.1f}%)")
                
                with col2:
                    if len(filtered_df) > 0:
                        event_spread = filtered_df.groupby('event_type').size()
                        st.info(f"Event Types Distribution: {', '.join([f'{k}: {v}' for k, v in event_spread.items()])}")
            else:
                st.warning("No data available for the selected filters")
        else:
            st.warning("Please select at least one event type")
    
    with tab3:
        # Top companies table
        st.markdown("#### Most Active Companies")
        
        n_companies = st.slider("Number of companies to display:", 5, 50, 15)
        top_companies = get_top_companies(df, n=n_companies)
        
        st.dataframe(
            top_companies,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button for the table
        csv = top_companies.to_csv(index=False)
        st.download_button(
            label="Download Top Companies CSV",
            data=csv,
            file_name=f"top_{n_companies}_companies.csv",
            mime="text/csv"
        )
    
    with tab4:
        # Data Explorer
        st.markdown("#### Raw Data Explorer")
        
        # Show filtered data with search
        search_term = st.text_input("Search in company names:", "")
        
        if search_term:
            explorer_df = filtered_df[filtered_df['label_name'].str.contains(search_term, case=False, na=False)]
        else:
            explorer_df = filtered_df
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(explorer_df):,}")
        with col2:
            st.metric("Columns", f"{len(explorer_df.columns)}")
        with col3:
            st.metric("Companies", f"{explorer_df['label_name'].nunique():,}")
        
        # Show data
        st.dataframe(explorer_df, use_container_width=True, height=400)
        
        # Download filtered data
        csv_filtered = explorer_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data CSV",
            data=csv_filtered,
            file_name="filtered_timeline_events.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(f"Data loaded: {len(df):,} events from {df['label_name'].nunique():,} companies")

if __name__ == "__main__":
    main()