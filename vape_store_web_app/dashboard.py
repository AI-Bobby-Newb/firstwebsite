import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from scipy import stats
from monthly_sales_analyzer import MonthlySalesAnalyzer

# Page configuration
st.set_page_config(
    page_title="Vape Store Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for responsive design
st.markdown("""
<style>
    /* Responsive adjustments for smaller screens */
    @media (max-width: 768px) {
        .main > div {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .stPlotlyChart {
            height: auto !important;
        }
        .st-emotion-cache-1r6slb0 {
            width: 100% !important;
        }
        /* Adjust metrics display */
        div[data-testid="stMetricValue"] > div {
            font-size: 1rem !important;
        }
        /* Adjust sidebar when collapsed */
        .st-emotion-cache-vz1v8s {
            min-width: unset !important;
        }
    }
    
    /* Better table scrolling on mobile */
    .dataframe-container {
        overflow-x: auto;
    }
    
    /* Improve spacing for mobile views */
    .responsive-container {
        padding: 0.2rem;
    }
    
    /* Make all charts scale properly */
    .plotly-chart-container {
        width: 100%;
        min-height: 300px;
    }
</style>
""", unsafe_allow_html=True)

# Define helper functions
def get_data_quality_score(df, reports):
    """Calculate a data quality score based on recency, completeness, and consistency"""
    score = 100  # Start with perfect score
    warnings = []
    
    # No data check
    if df.empty or len(reports) == 0:
        return 0, ["No data available"]
    
    # Check recency - how recent is the latest data?
    current_date = datetime.now()
    month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                   'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    # Get the most recent month and year from reports
    most_recent = None
    most_recent_year = 0
    most_recent_month = 0
    
    for report in reports:
        if isinstance(report, dict) and 'month' in report and 'year' in report:
            month_name = report['month']
            year = int(report['year'])
            month_num = month_order.get(month_name, 0)
            
            if year > most_recent_year or (year == most_recent_year and month_num > most_recent_month):
                most_recent_year = year
                most_recent_month = month_num
                most_recent = report
    
    # Check how recent the data is
    if most_recent:
        current_month = current_date.month
        current_year = current_date.year
        
        # Calculate months difference
        months_diff = (current_year - most_recent_year) * 12 + (current_month - most_recent_month)
        
        # Penalize for outdated data
        if months_diff > 3:
            score -= min(40, months_diff * 10)  # Max penalty of 40 points
            warnings.append(f"Data is {months_diff} months old")
        elif months_diff > 1:
            score -= months_diff * 5
            warnings.append(f"Data is {months_diff} months old")
    else:
        score -= 40
        warnings.append("Cannot determine data recency")
    
    # Check completeness - are there any gaps in the data?
    if len(reports) >= 2:
        # Sort reports by date
        def get_sort_key(report):
            if isinstance(report, dict) and 'month' in report and 'year' in report:
                month = report['month']
                year = int(report['year'])
                month_num = month_order.get(month, 0)
                return (year, month_num)
            return (0, 0)
        
        sorted_reports = sorted(reports, key=get_sort_key)
        
        # Check for gaps
        gaps = 0
        for i in range(1, len(sorted_reports)):
            prev_report = sorted_reports[i-1]
            curr_report = sorted_reports[i]
            
            if isinstance(prev_report, dict) and isinstance(curr_report, dict):
                prev_year = int(prev_report['year'])
                prev_month = month_order.get(prev_report['month'], 0)
                curr_year = int(curr_report['year'])
                curr_month = month_order.get(curr_report['month'], 0)
                
                # Calculate months difference
                months_diff = (curr_year - prev_year) * 12 + (curr_month - prev_month)
                
                # If difference is more than 1 month, there's a gap
                if months_diff > 1:
                    gaps += months_diff - 1
        
        if gaps > 0:
            score -= min(20, gaps * 5)  # Max penalty of 20 points
            warnings.append(f"{gaps} missing month(s) in data")
    
    # Check data consistency
    if not df.empty:
        # Check for unusually high variations
        monthly_sales = df.groupby('Month')['Net Sales'].sum().reset_index()
        if len(monthly_sales) >= 3:
            mean_sales = monthly_sales['Net Sales'].mean()
            std_sales = monthly_sales['Net Sales'].std()
            cv = std_sales / mean_sales if mean_sales > 0 else 0
            
            # If coefficient of variation is very high, flag it
            if cv > 0.5:  # Arbitrary threshold
                score -= min(20, int(cv * 20))  # Max penalty of 20 points
                warnings.append("High sales variability detected")
    
    # Normalize score to 0-100 range
    score = max(0, min(100, score))
    
    return score, warnings

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return MonthlySalesAnalyzer("monthly_sales")

analyzer = get_analyzer()

# Get reports data
@st.cache_data
def get_reports():
    reports = []
    for report_file in analyzer.get_available_reports():
        report = analyzer.parse_report(report_file)
        if report:
            reports.append(report)
    # Note: We're relying on the chronological ordering from analyzer.get_available_reports()
    # But still sorting here to ensure consistency
    return sorted(reports, key=lambda x: (x['year'], {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 
                                                     'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 
                                                     'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}.get(x['month'], 0)))

reports = get_reports()

# Create combined dataframe with all sales data
@st.cache_data
def get_combined_data(reports):
    all_dfs = []
    
    # Create month order mapping for sorting
    month_order = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                   'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    
    for report in reports:
        df = report['data'].copy()
        # Filter out TOTAL category
        df = df[df['Category Name'] != 'TOTAL']
        
        # Set month display format
        df['Month'] = f"{report['month']} {report['year']}"
        
        # Create sort key in YYYY-MM format for proper chronological sorting
        month_num = month_order.get(report['month'], '00')
        df['MonthSort'] = f"{report['year']}-{month_num}"
        
        # Store the numeric month value for additional sorting if needed
        df['MonthIndex'] = int(month_num)
        df['Year'] = int(report['year'])
        
        all_dfs.append(df)
    
    if all_dfs:
        return pd.concat(all_dfs)
    return pd.DataFrame()

combined_df = get_combined_data(reports)

# Calculate data quality score
quality_score, quality_warnings = get_data_quality_score(combined_df, reports)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Dashboard Overview", 
    "Monthly Analysis", 
    "Product Analysis", 
    "Category Analysis",
    "Top Products",
    "Advanced Analytics",
    "Business Insights"
])

# Display data quality indicator in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Quality")

# Determine color based on score
quality_color = "red" if quality_score < 50 else "orange" if quality_score < 75 else "green"

# Create a progress bar for the quality score
st.sidebar.markdown(f"<div style='display: flex; align-items: center;'>"
                   f"<div style='margin-right: 10px;'>Score:</div>"
                   f"<div style='flex-grow: 1; background-color: #f0f0f0; border-radius: 10px; height: 10px;'>"
                   f"<div style='width: {quality_score}%; background-color: {quality_color}; height: 10px; border-radius: 10px;'></div>"
                   f"</div>"
                   f"<div style='margin-left: 10px;'>{quality_score}/100</div>"
                   f"</div>", unsafe_allow_html=True)

# Show any warnings
if quality_warnings:
    with st.sidebar.expander("Data Quality Warnings", expanded=False):
        for warning in quality_warnings:
            st.warning(warning)

# Time filter in sidebar
st.sidebar.title("Time Filter")

# Custom function to get appropriate month-year ordering
def get_proper_month_order(month_data):
    # Extract the year and month from the month strings
    month_mapping = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    
    # Create a list of tuples: (year, month_num, original_month_name)
    ordered_months = []
    for month in month_data:
        parts = month.split()
        if len(parts) >= 2:
            month_name = parts[0]
            year = parts[1]
            month_num = month_mapping.get(month_name[:3], '00')
            ordered_months.append((year, month_num, month))
    
    # Sort by year, then month
    ordered_months.sort(key=lambda x: (x[0], x[1]))
    
    # Return just the original month names in the correct order
    return [item[2] for item in ordered_months]

# Get and sort months chronologically
months = combined_df['Month'].unique().tolist()
months = get_proper_month_order(months)

# Add a historical timeline display
st.sidebar.markdown("### Historical Timeline")
st.sidebar.markdown("Available Reports:")

# Group months by year for better organization
def group_months_by_year(months_list):
    years_dict = {}
    for month in months_list:
        parts = month.split()
        month_name = parts[0]
        year = parts[1]
        if year not in years_dict:
            years_dict[year] = []
        years_dict[year].append(month_name)
    return years_dict

# Display months organized by year
years_dict = group_months_by_year(months)
for year in sorted(years_dict.keys(), reverse=True):  # Newest years first
    month_names = years_dict[year]
    # Sort months within the year
    month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                   'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    sorted_months = sorted(month_names, key=lambda m: month_order.get(m, 0))
    month_list = ", ".join(sorted_months)
    st.sidebar.markdown(f"**{year}**: {month_list}")

# Create select all/none options
st.sidebar.markdown("### Filter Options")
all_option = st.sidebar.checkbox("Select All Months", True)

if all_option:
    selected_months = months
else:
    selected_months = st.sidebar.multiselect("Select Specific Months", months)

# Filter data based on selections
filtered_df = combined_df[combined_df['Month'].isin(selected_months)]

# Date range
min_date = filtered_df['MonthSort'].min() if not filtered_df.empty else ""
max_date = filtered_df['MonthSort'].max() if not filtered_df.empty else ""
date_range = f"{min_date} to {max_date}" if min_date and max_date else "All Time"

#---------------------------
# DASHBOARD OVERVIEW PAGE
#---------------------------
if page == "Dashboard Overview":
    st.title("Vape Store Sales Dashboard")
    st.subheader(f"Sales Overview - {date_range}")
    
    # Summary metrics row - responsive layout for smaller screens
    st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
    
    # Use 2 columns on small screens, 4 on larger screens
    use_container_width = True  # This will stretch metrics to use full available width
    metrics_per_row = 2
    
    total_sales = filtered_df['Net Sales'].sum()
    total_units = filtered_df['Sold'].sum()
    avg_price = total_sales / total_units if total_units > 0 else 0
    unique_products = filtered_df['Name'].nunique()
    
    # Create rows dynamically for better mobile layout
    metric_row1 = st.columns(metrics_per_row)
    metric_row2 = st.columns(metrics_per_row)
    
    metric_row1[0].metric("Total Sales", f"${total_sales:,.2f}")
    metric_row1[1].metric("Units Sold", f"{total_units:,}")
    metric_row2[0].metric("Average Price", f"${avg_price:.2f}")
    metric_row2[1].metric("Unique Products", f"{unique_products:,}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts row with responsive layout
    st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
    
    # Use a single column on mobile, two columns on larger screens
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("Monthly Sales Trend")
        st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
        
        monthly_sales = filtered_df.groupby(['Month', 'MonthSort'])['Net Sales'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values('MonthSort')
        
        # Get list of months in proper chronological order for x-axis
        month_order = get_proper_month_order(monthly_sales['Month'].unique().tolist())
        
        fig = px.bar(
            monthly_sales, 
            x='Month', 
            y='Net Sales',
            title="Monthly Sales",
            labels={'Net Sales': 'Net Sales ($)', 'Month': ''},
            color_discrete_sequence=['#1f77b4'],
            text_auto='.2s',
            category_orders={"Month": month_order}  # Explicitly set the order of months
        )
        
        # Configure layout for better mobile display
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': month_order},
            xaxis_tickangle=-45,
            margin=dict(l=10, r=10, t=30, b=40),  # Tighter margins
            height=350,                          # Fixed height for consistency
            autosize=True,                       # Allow chart to be responsive
            font=dict(size=10)                  # Smaller font for mobile
        )
        
        # Make text responsive
        fig.update_traces(
            textfont=dict(size=10),
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Top Categories")
        st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
        
        category_sales = filtered_df.groupby('Category Name')['Net Sales'].sum().reset_index()
        category_sales = category_sales.sort_values('Net Sales', ascending=False).head(5)
        
        fig = px.pie(
            category_sales, 
            values='Net Sales', 
            names='Category Name',
            title="Sales by Category",
            hole=0.4
        )
        
        # Configure layout for better mobile display
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Tighter margins
            height=350,                           # Fixed height for consistency
            autosize=True,                        # Allow chart to be responsive
            font=dict(size=10),                   # Smaller font for mobile
            legend=dict(
                orientation="h",                  # Horizontal legend
                yanchor="bottom",
                y=-0.1,                          # Position below chart
                xanchor="center",
                x=0.5
            )
        )
        
        # Simplify labels for mobile
        fig.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont_size=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close previous responsive container and start a new one for bottom row
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
    
    # Bottom row charts - use a single column on mobile, two columns on larger screens
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("Top 10 Products by Revenue")
        st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
        
        top_products = filtered_df.groupby('Name')['Net Sales'].sum().reset_index()
        top_products = top_products.sort_values('Net Sales', ascending=False).head(10)
        
        # Truncate long product names for better mobile display
        top_products['Name'] = top_products['Name'].apply(lambda x: x[:20] + '...' if len(str(x)) > 20 else x)
        
        fig = px.bar(
            top_products,
            x='Net Sales',
            y='Name',
            orientation='h',
            title="Top 10 Products by Revenue",
            labels={'Net Sales': 'Net Sales ($)', 'Name': ''},
            text_auto='.2s'
        )
        
        # Configure layout for better mobile display
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=10, r=10, t=30, b=10),  # Tighter margins
            height=400,                           # Fixed height for consistency
            autosize=True,                        # Allow chart to be responsive
            font=dict(size=10)                    # Smaller font for mobile
        )
        
        # Make text responsive
        fig.update_traces(
            textfont=dict(size=9),
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Top 10 Products by Units Sold")
        st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
        
        top_units = filtered_df.groupby('Name')['Sold'].sum().reset_index()
        top_units = top_units.sort_values('Sold', ascending=False).head(10)
        
        # Truncate long product names for better mobile display
        top_units['Name'] = top_units['Name'].apply(lambda x: x[:20] + '...' if len(str(x)) > 20 else x)
        
        fig = px.bar(
            top_units,
            x='Sold',
            y='Name',
            orientation='h',
            title="Top 10 Products by Units Sold",
            labels={'Sold': 'Units Sold', 'Name': ''},
            text_auto='.2s',
            color_discrete_sequence=['#2ca02c']
        )
        
        # Configure layout for better mobile display
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=10, r=10, t=30, b=10),  # Tighter margins
            height=400,                           # Fixed height for consistency
            autosize=True,                        # Allow chart to be responsive
            font=dict(size=10)                    # Smaller font for mobile
        )
        
        # Make text responsive
        fig.update_traces(
            textfont=dict(size=9),
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the responsive container
    st.markdown('</div>', unsafe_allow_html=True)

#---------------------------
# MONTHLY ANALYSIS PAGE
#---------------------------
elif page == "Monthly Analysis":
    st.title("Monthly Sales Analysis")
    st.subheader(f"Analysis Period: {date_range}")
    
    # Monthly comparison with proper sorting
    monthly_data = filtered_df.groupby(['Month', 'MonthSort', 'Year', 'MonthIndex']).agg({
        'Net Sales': 'sum',
        'Sold': 'sum'
    }).reset_index()
    
    # Sort by MonthSort for chronological ordering
    monthly_data = monthly_data.sort_values('MonthSort')
    
    # Calculate month-over-month growth
    if len(monthly_data) > 1:
        monthly_data['Sales Growth'] = monthly_data['Net Sales'].pct_change() * 100
        monthly_data['Units Growth'] = monthly_data['Sold'].pct_change() * 100
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Net Sales by Month")
        # Create a list of month names in the correct chronological order
        month_order = get_proper_month_order(monthly_data['Month'].unique().tolist())
        
        fig = px.line(
            monthly_data,
            x='Month',
            y='Net Sales',
            markers=True,
            title="Monthly Sales Trend",
            labels={'Net Sales': 'Net Sales ($)', 'Month': ''}
        )
        # Force the x-axis to respect the month order
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': month_order})
        # Add data points labels
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='top center')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Units Sold by Month")
        # Create a list of month names in the correct chronological order
        month_order = get_proper_month_order(monthly_data['Month'].unique().tolist())
        
        fig = px.line(
            monthly_data,
            x='Month',
            y='Sold',
            markers=True,
            title="Monthly Units Sold",
            labels={'Sold': 'Units Sold', 'Month': ''},
            color_discrete_sequence=['#2ca02c']
        )
        # Force the x-axis to respect the month order
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': month_order})
        # Add data points labels
        fig.update_traces(texttemplate='%{y}', textposition='top center')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate chart if enough data
    if len(monthly_data) > 1 and 'Sales Growth' in monthly_data.columns:
        st.subheader("Month-over-Month Growth")
        growth_df = monthly_data.copy()
        growth_df = growth_df.dropna(subset=['Sales Growth'])
        
        # Ensure proper ordering of months
        month_order = get_proper_month_order(monthly_data['Month'].unique().tolist())
        growth_df = growth_df.sort_values('MonthSort')  # Sort by month
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=growth_df['Month'],
            y=growth_df['Sales Growth'],
            name='Sales Growth %',
            marker_color=['red' if x < 0 else 'green' for x in growth_df['Sales Growth']]
        ))
        
        # Force the x-axis to respect the month order
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': month_order})
        
        fig.update_layout(
            title="Month-over-Month Sales Growth (%)",
            xaxis_title="",
            yaxis_title="Growth %",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly performance table
    st.subheader("Monthly Performance")
    display_df = monthly_data.copy()
    
    # Ensure MonthSort is properly formatted for sorting (YYYY-MM format)
    if 'MonthSort' in display_df.columns:
        # Sort chronologically by MonthSort before displaying
        display_df = display_df.sort_values('MonthSort')
        
        # Create a proper index for Streamlit to respect the order
        display_df = display_df.reset_index(drop=True)
    
    # Format columns for display
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.2f}")
    display_df['Sold'] = display_df['Sold'].apply(lambda x: f"{x:,}")
    
    if 'Sales Growth' in display_df.columns:
        display_df['Sales Growth'] = display_df['Sales Growth'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Select and reorder columns for display
    columns_to_display = ['Month', 'Net Sales', 'Sold']
    if 'Sales Growth' in display_df.columns:
        columns_to_display.append('Sales Growth')
    
    # Wrap dataframe in a scrollable container for mobile
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    
    # Convert to a simpler DataFrame that Streamlit will display in the exact order
    ordered_df = pd.DataFrame(
        {col: display_df[col].tolist() for col in columns_to_display},
        index=range(len(display_df))
    )
    
    # Display the ordered dataframe
    st.dataframe(ordered_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

#---------------------------
# PRODUCT ANALYSIS PAGE
#---------------------------
elif page == "Product Analysis":
    st.title("Product Performance Analysis")
    st.subheader(f"Analysis Period: {date_range}")
    
    # Filter options for products with responsive layout
    st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
    
    # Use a single column on very small screens
    col1, col2 = st.columns([3, 2], gap="small")
    
    with col1:
        # Get unique categories
        categories = sorted(filtered_df['Category Name'].unique().tolist())
        selected_category = st.selectbox("Filter by Category", ["All Categories"] + categories)
    
    # Apply category filter
    if selected_category != "All Categories":
        product_df = filtered_df[filtered_df['Category Name'] == selected_category]
    else:
        product_df = filtered_df
    
    # Product search box
    search_term = st.text_input("Search Products", "")
    
    if search_term:
        product_df = product_df[product_df['Name'].str.contains(search_term, case=False, na=False)]
    
    # Product performance metrics
    st.subheader("Product Performance")
    
    # Group by product
    product_perf = product_df.groupby('Name').agg({
        'Net Sales': 'sum',
        'Sold': 'sum',
        'Category Name': 'first'  # Get the first category for each product
    }).reset_index()
    
    # Calculate average price
    product_perf['Avg Price'] = product_perf['Net Sales'] / product_perf['Sold']
    
    # Sort by revenue (default)
    product_perf = product_perf.sort_values('Net Sales', ascending=False)
    
    # Top 10 products visualization
    top_products = product_perf.head(10)
    
    if not top_products.empty:
        # Wrap chart in container for responsiveness
        st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
        
        # Truncate long product names for better mobile display
        top_products_display = top_products.copy()
        top_products_display['Name'] = top_products_display['Name'].apply(lambda x: x[:15] + '...' if len(str(x)) > 15 else x)
        
        # Bar chart for top products
        fig = px.bar(
            top_products_display,
            x='Name',
            y='Net Sales',
            color='Category Name',
            title=f"Top 10 Products by Revenue {f'in {selected_category}' if selected_category != 'All Categories' else ''}",
            labels={'Net Sales': 'Net Sales ($)', 'Name': '', 'Category Name': 'Category'},
            text_auto='.2s'
        )
        
        # Configure layout for better mobile display
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=10, r=10, t=40, b=80),  # More bottom margin for rotated labels
            height=400,                           # Fixed height for consistency
            autosize=True,                        # Allow chart to be responsive
            font=dict(size=10),                   # Smaller font for mobile
            legend=dict(
                orientation="h",                  # Horizontal legend
                yanchor="bottom",
                y=-0.4,                          # Position below chart
                xanchor="center",
                x=0.5
            )
        )
        
        # Make text responsive
        fig.update_traces(
            textfont=dict(size=9),
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Scatter plot showing units sold vs revenue
        st.subheader("Units Sold vs Revenue")
        st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
        
        scatter_df = product_perf[product_perf['Sold'] > 0].copy()  # Exclude products with 0 sales
        
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x='Sold',
                y='Net Sales',
                color='Category Name',
                size='Avg Price',
                hover_name='Name',
                title="Product Performance Map",
                labels={
                    'Net Sales': 'Net Sales ($)',
                    'Sold': 'Units Sold',
                    'Category Name': 'Category',
                    'Avg Price': 'Avg Price ($)'
                }
            )
            
            # Configure layout for better mobile display
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),  # Tighter margins
                height=400,                           # Fixed height for consistency
                autosize=True,                        # Allow chart to be responsive
                font=dict(size=10),                   # Smaller font for mobile
                legend=dict(
                    orientation="h",                  # Horizontal legend
                    yanchor="bottom",
                    y=-0.15,                         # Position below chart
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display product table
    st.subheader("Product Details")
    
    # Format the data for display
    display_df = product_perf.copy()
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.2f}")
    display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)

#---------------------------
# CATEGORY ANALYSIS PAGE
#---------------------------
elif page == "Category Analysis":
    st.title("Category Analysis")
    st.subheader(f"Analysis Period: {date_range}")
    
    # Calculate category metrics
    category_data = filtered_df.groupby('Category Name').agg({
        'Net Sales': 'sum',
        'Sold': 'sum',
        'Name': pd.Series.nunique  # Count unique products in each category
    }).reset_index()
    
    # Calculate percentage of total sales
    total_sales = category_data['Net Sales'].sum()
    category_data['Percent of Sales'] = (category_data['Net Sales'] / total_sales * 100)
    
    # Calculate average price per unit
    category_data['Avg Price'] = category_data['Net Sales'] / category_data['Sold']
    
    # Sort by revenue
    category_data = category_data.sort_values('Net Sales', ascending=False)
    
    # Rename Name column to Products
    category_data = category_data.rename(columns={'Name': 'Products'})
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Sales Distribution")
        fig = px.pie(
            category_data, 
            values='Net Sales',
            names='Category Name',
            title="Sales by Category",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Categories by Revenue")
        top_categories = category_data.head(10)
        
        fig = px.bar(
            top_categories,
            x='Category Name',
            y='Net Sales',
            color='Category Name',
            title="Revenue by Category",
            labels={'Net Sales': 'Net Sales ($)', 'Category Name': ''},
            text_auto='.2s'
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Category metrics over time
    st.subheader("Category Performance Over Time")
    
    # Get top 5 categories by sales
    top_categories = category_data.head(5)['Category Name'].tolist()
    selected_categories = st.multiselect(
        "Select Categories to Compare",
        options=category_data['Category Name'].tolist(),
        default=top_categories
    )
    
    if selected_categories:
        # Filter data for selected categories and group by month
        cat_time_data = filtered_df[filtered_df['Category Name'].isin(selected_categories)]
        cat_trend = cat_time_data.groupby(['Month', 'Category Name']).agg({
            'Net Sales': 'sum'
        }).reset_index()
        
        # Create line chart
        # Get list of months in proper chronological order for x-axis
        month_order = get_proper_month_order(filtered_df['Month'].unique().tolist())
        
        fig = px.line(
            cat_trend,
            x='Month',
            y='Net Sales',
            color='Category Name',
            markers=True,
            title="Category Sales Trends Over Time",
            labels={'Net Sales': 'Net Sales ($)', 'Month': ''},
            category_orders={"Month": month_order}  # Explicitly set the order of months
        )
        
        # Force the x-axis to respect the month order
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': month_order})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display category table
    st.subheader("Category Details")
    
    # Format the data for display
    display_df = category_data.copy()
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.2f}")
    display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df['Percent of Sales'] = display_df['Percent of Sales'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(display_df, use_container_width=True)

#---------------------------
# TOP PRODUCTS PAGE
#---------------------------
elif page == "Top Products":
    st.title("Top Products Analysis")
    st.subheader(f"Analysis Period: {date_range}")
    
    # Tab layout for different rankings
    tab1, tab2, tab3 = st.tabs(["By Revenue", "By Units Sold", "By Profit Margin"])
    
    # Get product metrics
    product_data = filtered_df.groupby(['Name', 'Category Name']).agg({
        'Net Sales': 'sum',
        'Sold': 'sum'
    }).reset_index()
    
    # Calculate average price
    product_data['Avg Price'] = product_data['Net Sales'] / product_data['Sold']
    
    with tab1:
        st.subheader("Top Products by Revenue")
        
        # Number of products to show
        top_n = st.slider("Number of products to display", 5, 50, 20, key="revenue_slider")
        
        # Get top products by revenue
        top_by_rev = product_data.sort_values('Net Sales', ascending=False).head(top_n)
        
        # Bar chart for top products
        fig = px.bar(
            top_by_rev,
            y='Name',
            x='Net Sales',
            color='Category Name',
            orientation='h',
            title=f"Top {top_n} Products by Revenue",
            labels={'Net Sales': 'Net Sales ($)', 'Name': '', 'Category Name': 'Category'},
            text_auto='.2s'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table with top products
        display_df = top_by_rev.copy()
        display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.2f}")
        display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.subheader("Top Products by Units Sold")
        
        # Number of products to show
        top_n = st.slider("Number of products to display", 5, 50, 20, key="units_slider")
        
        # Get top products by units sold
        top_by_units = product_data.sort_values('Sold', ascending=False).head(top_n)
        
        # Bar chart for top products
        fig = px.bar(
            top_by_units,
            y='Name',
            x='Sold',
            color='Category Name',
            orientation='h',
            title=f"Top {top_n} Products by Units Sold",
            labels={'Sold': 'Units Sold', 'Name': '', 'Category Name': 'Category'},
            text_auto='.2s'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table with top products
        display_df = top_by_units.copy()
        display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.2f}")
        display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        st.subheader("Top Products by Average Price")
        
        # Number of products to show
        top_n = st.slider("Number of products to display", 5, 50, 20, key="margin_slider")
        
        # Minimum units sold filter to avoid outliers
        min_units = st.slider("Minimum units sold", 1, 50, 5)
        
        # Filter products with enough units sold and sort by average price
        top_by_margin = product_data[product_data['Sold'] >= min_units].sort_values('Avg Price', ascending=False).head(top_n)
        
        # Bar chart for top products
        fig = px.bar(
            top_by_margin,
            y='Name',
            x='Avg Price',
            color='Category Name',
            orientation='h',
            title=f"Top {top_n} Products by Average Price (min {min_units} units sold)",
            labels={'Avg Price': 'Average Price ($)', 'Name': '', 'Category Name': 'Category'},
            text_auto='.2s'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table with top products
        display_df = top_by_margin.copy()
        display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.2f}")
        display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)

#---------------------------
# ADVANCED ANALYTICS PAGE
#---------------------------
elif page == "Advanced Analytics":
    st.title("Advanced Sales Analytics")
    st.subheader(f"Analysis Period: {date_range}")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Product Velocity", "Price Analysis", "Trend Metrics", "Anomaly Detection", "Market Basket"])
    
    with tab1:
        st.subheader("Product Sales Velocity Analysis")
        st.write("This analysis shows how quickly products are selling over time.")
        
        # Calculate days in the analysis period
        if not filtered_df.empty:
            # Convert MonthSort to datetime for date calculations
            date_data = filtered_df[['MonthSort']].drop_duplicates()
            
            # Handle month numbers by replacing '00' with '01' to avoid date parsing errors
            date_data['MonthSort_fixed'] = date_data['MonthSort'].str.replace('-00', '-01')
            date_data['Date'] = pd.to_datetime(date_data['MonthSort_fixed'], format='%Y-%m', errors='coerce')
            
            # Drop any rows with NaT values that might have resulted from parsing errors
            date_data = date_data.dropna(subset=['Date'])
            
            # Calculate date range in days, with minimum of 30 days
            if len(date_data) >= 2:
                date_range_days = max((date_data['Date'].max() - date_data['Date'].min()).days, 30)
            else:
                date_range_days = 30  # Default to 30 days if only one month or parsing failed
            
            # Calculate sales velocity (units sold per day)
            product_velocity = filtered_df.groupby(['Name', 'Category Name']).agg({
                'Sold': 'sum',
                'Net Sales': 'sum'
            }).reset_index()
            
            product_velocity['Sales Velocity'] = product_velocity['Sold'] / date_range_days
            product_velocity['Revenue Velocity'] = product_velocity['Net Sales'] / date_range_days
            
            # Sort by velocity
            top_velocity = product_velocity.sort_values('Sales Velocity', ascending=False).head(15)
            
            # Create velocity chart
            fig = px.bar(
                top_velocity,
                y='Name',
                x='Sales Velocity',
                color='Category Name',
                orientation='h',
                title=f"Top 15 Products by Sales Velocity (Units/Day)",
                labels={
                    'Sales Velocity': 'Units Sold per Day',
                    'Name': '',
                    'Category Name': 'Category'
                }
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show velocity metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Product Velocity", f"{product_velocity['Sales Velocity'].mean():.2f} units/day")
            with col2:
                st.metric("Median Product Velocity", f"{product_velocity['Sales Velocity'].median():.2f} units/day")
            
            # Revenue velocity
            st.subheader("Revenue Velocity Analysis")
            top_rev_velocity = product_velocity.sort_values('Revenue Velocity', ascending=False).head(15)
            
            fig = px.bar(
                top_rev_velocity,
                y='Name',
                x='Revenue Velocity',
                color='Category Name',
                orientation='h',
                title=f"Top 15 Products by Revenue Velocity ($/Day)",
                labels={
                    'Revenue Velocity': 'Revenue per Day ($)',
                    'Name': '',
                    'Category Name': 'Category'
                }
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data available for the selected period.")
    
    with tab2:
        st.subheader("Price Point Analysis")
        st.write("This analysis examines sales performance across different price points.")
        
        if not filtered_df.empty:
            # Calculate average price for each product
            price_analysis = filtered_df.groupby('Name').agg({
                'Net Sales': 'sum',
                'Sold': 'sum',
                'Category Name': 'first'
            }).reset_index()
            
            price_analysis['Avg Price'] = price_analysis['Net Sales'] / price_analysis['Sold']
            
            # Create price bins
            price_analysis['Price Range'] = pd.cut(
                price_analysis['Avg Price'],
                bins=[0, 10, 20, 30, 40, 50, 100, 1000],
                labels=['$0-10', '$10-20', '$20-30', '$30-40', '$40-50', '$50-100', '$100+'],
                right=False
            )
            
            # Aggregate by price range
            price_range_analysis = price_analysis.groupby('Price Range').agg({
                'Name': 'count',
                'Net Sales': 'sum',
                'Sold': 'sum'
            }).reset_index()
            
            price_range_analysis = price_range_analysis.rename(columns={'Name': 'Product Count'})
            price_range_analysis['Avg Price in Range'] = price_range_analysis['Net Sales'] / price_range_analysis['Sold']
            
            # Create charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Product count by price range
                fig = px.bar(
                    price_range_analysis,
                    x='Price Range',
                    y='Product Count',
                    title="Number of Products by Price Range",
                    labels={
                        'Price Range': 'Price Range',
                        'Product Count': 'Number of Products'
                    },
                    color='Product Count'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue by price range
                fig = px.bar(
                    price_range_analysis,
                    x='Price Range',
                    y='Net Sales',
                    title="Revenue by Price Range",
                    labels={
                        'Price Range': 'Price Range',
                        'Net Sales': 'Revenue ($)'
                    },
                    color='Net Sales'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Units sold by price range
            fig = px.bar(
                price_range_analysis,
                x='Price Range',
                y='Sold',
                title="Units Sold by Price Range",
                labels={
                    'Price Range': 'Price Range',
                    'Sold': 'Units Sold'
                },
                color='Sold'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Product quadrant analysis (Price vs. Volume)
            st.subheader("Product Performance Quadrant")
            st.write("This chart plots products by price and sales volume to identify high-value products.")
            
            # Calculate median values for quadrant lines
            median_price = price_analysis['Avg Price'].median()
            median_sold = price_analysis['Sold'].median()
            
            fig = px.scatter(
                price_analysis,
                x='Avg Price',
                y='Sold',
                color='Category Name',
                size='Net Sales',
                hover_name='Name',
                log_x=True,  # Log scale for price to better visualize distribution
                title="Product Performance Quadrant: Price vs. Volume",
                labels={
                    'Avg Price': 'Average Price ($)',
                    'Sold': 'Units Sold',
                    'Category Name': 'Category'
                }
            )
            
            # Add quadrant lines
            fig.add_hline(y=median_sold, line_dash="dash", line_color="gray")
            fig.add_vline(x=median_price, line_dash="dash", line_color="gray")
            
            # Add quadrant annotations
            fig.add_annotation(x=median_price/2, y=median_sold*1.5, text="Low Price, High Volume", showarrow=False)
            fig.add_annotation(x=median_price*2, y=median_sold*1.5, text="Premium Products", showarrow=False)
            fig.add_annotation(x=median_price/2, y=median_sold/2, text="Low Performers", showarrow=False)
            fig.add_annotation(x=median_price*2, y=median_sold/2, text="Luxury Products", showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data available for the selected period.")
    
    with tab3:
        st.subheader("Sales Trend Metrics")
        
        if not filtered_df.empty and len(months) > 1:
            # Monthly trend with moving average
            monthly_sales_trend = filtered_df.groupby(['Month', 'MonthSort', 'Year', 'MonthIndex']).agg({
                'Net Sales': 'sum',
                'Sold': 'sum'
            }).reset_index()
            
            # IMPORTANT: Sort chronologically by MonthSort before calculating rolling average
            monthly_sales_trend = monthly_sales_trend.sort_values('MonthSort')
            
            # Calculate moving averages if we have enough data points
            if len(monthly_sales_trend) >= 3:
                # First, ensure data is properly sorted by MonthSort
                # Use min_periods=1 to handle edge cases at the beginning of series
                monthly_sales_trend['Sales 3-Month MA'] = monthly_sales_trend['Net Sales'].rolling(window=3, min_periods=1).mean()
                
                # Fill any remaining NaN values with the actual sales value
                # This ensures the trend line starts from the first data point
                mask = monthly_sales_trend['Sales 3-Month MA'].isna()
                monthly_sales_trend.loc[mask, 'Sales 3-Month MA'] = monthly_sales_trend.loc[mask, 'Net Sales']
                
                # Add simple forecast for next 3 months
                if len(monthly_sales_trend) >= 6:  # Need enough data for a meaningful forecast
                    # Get the last 6 data points for forecasting
                    forecast_data = monthly_sales_trend.tail(6).copy()
                    
                    # Create a numeric index for regression
                    forecast_data['index'] = range(len(forecast_data))
                    
                    # Fit a simple linear regression model
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        forecast_data['index'], forecast_data['Net Sales']
                    )
                    
                    # Create forecast points for next 3 months
                    last_idx = forecast_data['index'].iloc[-1]
                    next_indices = [last_idx + i + 1 for i in range(3)]
                    
                    # Get last month details to create new month labels
                    last_month = forecast_data['Month'].iloc[-1]
                    last_month_parts = last_month.split()
                    if len(last_month_parts) >= 2:
                        last_month_name = last_month_parts[0]
                        last_year = int(last_month_parts[1])
                        
                        # Month order for generating next months
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        # Find index of last month
                        try:
                            last_month_idx = month_names.index(last_month_name)
                            
                            # Generate next 3 month names
                            forecast_months = []
                            for i in range(3):
                                next_month_idx = (last_month_idx + i + 1) % 12
                                next_year = last_year + (1 if next_month_idx <= last_month_idx else 0)
                                forecast_months.append(f"{month_names[next_month_idx]} {next_year}")
                                
                            # Calculate forecast values
                            forecast_values = [intercept + slope * idx for idx in next_indices]
                            
                            # Ensure the main dataframe has a Forecast column
                            if 'Forecast' not in monthly_sales_trend.columns:
                                monthly_sales_trend['Forecast'] = False
                            
                            # Add seasonality adjustment to forecasting if we have enough data
                            if len(monthly_sales_trend) >= 12:  # Need at least a year of data for seasonality
                                # Extract month numbers for seasonality analysis
                                monthly_sales_trend['MonthNum'] = monthly_sales_trend['Month'].apply(
                                    lambda x: month_names.index(x.split()[0]) + 1 if x.split()[0] in month_names else 0
                                )
                                
                                # Calculate average seasonal factors
                                seasonal_factors = {}
                                for month_num in range(1, 13):  # 1-12 for Jan-Dec
                                    month_data = monthly_sales_trend[monthly_sales_trend['MonthNum'] == month_num]
                                    if not month_data.empty:
                                        avg_sales = month_data['Net Sales'].mean()
                                        # Calculate seasonal factor relative to overall average
                                        overall_avg = monthly_sales_trend['Net Sales'].mean()
                                        seasonal_factors[month_num] = avg_sales / overall_avg if overall_avg > 0 else 1.0
                                    else:
                                        seasonal_factors[month_num] = 1.0  # Default factor
                                
                                # Apply seasonal adjustments to forecast values
                                adjusted_forecast_values = []
                                for i, base_value in enumerate(forecast_values):
                                    next_month_num = (last_month_idx + i + 1) % 12 + 1  # Convert to 1-12 range
                                    seasonal_factor = seasonal_factors.get(next_month_num, 1.0)
                                    # Apply seasonal factor to linear forecast
                                    adjusted_value = base_value * seasonal_factor
                                    adjusted_forecast_values.append(max(0, adjusted_value))  # Ensure no negative values
                                
                                # Use seasonally adjusted values if available
                                forecast_values = adjusted_forecast_values
                            else:
                                # Ensure no negative values in the original forecast
                                forecast_values = [max(0, val) for val in forecast_values]
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'Month': forecast_months,
                                'Net Sales': forecast_values,
                                'Sales 3-Month MA': forecast_values,  # Same values for MA
                                'Forecast': True
                            })
                            
                            # Make sure monthly_sales_trend has all necessary columns
                            # before marking existing data as not forecast
                            if 'Forecast' not in monthly_sales_trend.columns:
                                monthly_sales_trend['Forecast'] = False
                            else:
                                # Mark existing data as not forecast if column already exists
                                monthly_sales_trend['Forecast'] = False
                            
                            # Append forecast to original data
                            monthly_sales_trend = pd.concat([monthly_sales_trend, forecast_df])
                        except ValueError:
                            # If month parsing fails, skip forecast
                            pass
                
                # Create month order for proper chronological display
                month_order = get_proper_month_order(monthly_sales_trend['Month'].unique().tolist())
                
                # Plot trend with moving average
                fig = go.Figure()
                
                # Check if 'Forecast' column exists
                if 'Forecast' in monthly_sales_trend.columns:
                    # Split data into actual and forecast parts
                    actual_data = monthly_sales_trend[~monthly_sales_trend['Forecast']]
                    forecast_data = monthly_sales_trend[monthly_sales_trend['Forecast']]
                else:
                    # If no forecast column, all data is actual
                    actual_data = monthly_sales_trend
                    forecast_data = pd.DataFrame(columns=monthly_sales_trend.columns)
                
                # Plot actual data
                fig.add_trace(go.Scatter(
                    x=actual_data['Month'],
                    y=actual_data['Net Sales'],
                    mode='lines+markers',
                    name='Monthly Sales',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8),
                    hovertemplate='%{y:$.2f}'
                ))
                
                # Plot moving average for actual data
                fig.add_trace(go.Scatter(
                    x=actual_data['Month'],
                    y=actual_data['Sales 3-Month MA'],
                    mode='lines',
                    name='3-Month Moving Average',
                    line=dict(color='red', dash='dash', width=2, shape='spline'),
                    hovertemplate='%{y:$.2f}'
                ))
                
                # Plot forecast data if available and column exists
                if 'Forecast' in monthly_sales_trend.columns and not forecast_data.empty:
                    # Add forecast data points
                    fig.add_trace(go.Scatter(
                        x=forecast_data['Month'],
                        y=forecast_data['Net Sales'],
                        mode='markers',
                        name='Sales Forecast',
                        marker=dict(size=8, symbol='diamond', color='green'),
                        hovertemplate='Forecast: %{y:$.2f}'
                    ))
                    
                    # Add forecast trend line
                    fig.add_trace(go.Scatter(
                        x=forecast_data['Month'],
                        y=forecast_data['Net Sales'],
                        mode='lines',
                        name='Forecast Trend',
                        line=dict(color='green', dash='dot', width=2),
                        hovertemplate='Forecast: %{y:$.2f}'
                    ))
                
                # Force x-axis to respect chronological order
                fig.update_layout(
                    title="Sales Trend with 3-Month Moving Average",
                    xaxis_title="",
                    yaxis_title="Sales ($)",
                    xaxis={'categoryorder': 'array', 'categoryarray': month_order},
                    xaxis_tickangle=-45,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified",
                    margin=dict(t=50, b=50),
                    plot_bgcolor='white',
                    yaxis=dict(
                        gridcolor='lightgray',
                        zerolinecolor='lightgray'
                    )
                )
                
                # Add annotations showing the trend direction and seasonality
                if len(monthly_sales_trend) >= 2:
                    # For trend direction
                    actual_data = monthly_sales_trend[~monthly_sales_trend.get('Forecast', False)]
                    first_val = actual_data['Net Sales'].iloc[0]
                    last_val = actual_data['Net Sales'].iloc[-1]
                    change_pct = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                    
                    direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
                    color = "green" if change_pct > 0 else "red" if change_pct < 0 else "gray"
                    
                    fig.add_annotation(
                        x=0.02,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"Overall Trend: {direction} {abs(change_pct):.1f}%",
                        showarrow=False,
                        font=dict(size=14, color=color),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=color,
                        borderwidth=1,
                        borderpad=4
                    )
                    
                    # For seasonality detection
                    if len(actual_data) >= 12:  # Need at least a year of data
                        # Create month mapping for analysis
                        month_mapping = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        
                        # Extract month numbers for seasonality analysis
                        actual_data['MonthNum'] = actual_data['Month'].apply(
                            lambda x: month_mapping.get(x.split()[0][:3], 0) if isinstance(x, str) else 0
                        )
                        
                        # Group by month number and calculate average sales
                        month_groups = actual_data.groupby('MonthNum')['Net Sales'].mean().reset_index()
                        
                        if len(month_groups) > 1:
                            # Calculate coefficient of variation to detect seasonality
                            mean_sales = month_groups['Net Sales'].mean()
                            std_sales = month_groups['Net Sales'].std()
                            cv = (std_sales / mean_sales) if mean_sales > 0 else 0
                            
                            # Find peak and trough months
                            peak_month = month_groups.loc[month_groups['Net Sales'].idxmax()]
                            trough_month = month_groups.loc[month_groups['Net Sales'].idxmin()]
                            
                            # Convert month numbers back to names
                            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            peak_month_name = month_names[int(peak_month['MonthNum']) - 1] if 1 <= int(peak_month['MonthNum']) <= 12 else ''
                            trough_month_name = month_names[int(trough_month['MonthNum']) - 1] if 1 <= int(trough_month['MonthNum']) <= 12 else ''
                            
                            # Determine if there's significant seasonality (CV > 0.15 is a common threshold)
                            if cv > 0.15 and peak_month_name and trough_month_name:
                                max_min_diff = peak_month['Net Sales'] - trough_month['Net Sales']
                                diff_pct = (max_min_diff / trough_month['Net Sales'] * 100) if trough_month['Net Sales'] > 0 else 0
                                
                                fig.add_annotation(
                                    x=0.02,
                                    y=0.87,
                                    xref="paper",
                                    yref="paper",
                                    text=f"Seasonal Pattern Detected: Peak in {peak_month_name}, Trough in {trough_month_name}",
                                    showarrow=False,
                                    font=dict(size=13, color="purple"),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="purple",
                                    borderwidth=1,
                                    borderpad=4
                                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cumulative sales analysis
            monthly_sales_trend['Cumulative Sales'] = monthly_sales_trend['Net Sales'].cumsum()
            monthly_sales_trend['Cumulative Units'] = monthly_sales_trend['Sold'].cumsum()
            
            # Plot cumulative sales
            fig = px.line(
                monthly_sales_trend,
                x='Month',
                y='Cumulative Sales',
                markers=True,
                title="Cumulative Sales Over Time",
                labels={'Cumulative Sales': 'Cumulative Sales ($)', 'Month': ''}
            )
            # Force x-axis order
            fig.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': month_order},
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Year-over-year comparison if we have data from multiple years
            years = filtered_df['Year'].unique()
            if len(years) > 1:
                st.subheader("Year-over-Year Comparison")
                
                # Group by month and year
                yoy_data = filtered_df.groupby(['Year', 'MonthIndex']).agg({
                    'Net Sales': 'sum',
                    'Sold': 'sum'
                }).reset_index()
                
                # Pivot table for YoY comparison
                yoy_sales = yoy_data.pivot(index='MonthIndex', columns='Year', values='Net Sales').reset_index()
                
                # Replace month index with month names
                month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                yoy_sales['Month'] = yoy_sales['MonthIndex'].map(month_names)
                
                # Create YoY comparison chart
                fig = go.Figure()
                
                for year in sorted(years):
                    if year in yoy_sales.columns:
                        fig.add_trace(go.Scatter(
                            x=yoy_sales['Month'],
                            y=yoy_sales[year],
                            mode='lines+markers',
                            name=str(year)
                        ))
                
                fig.update_layout(
                    title="Year-over-Year Sales Comparison",
                    xaxis_title="",
                    yaxis_title="Sales ($)",
                    xaxis={'categoryorder': 'array', 'categoryarray': list(month_names.values())},
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for trend analysis. Please select more months or years.")
    
    with tab4:
        st.subheader("Anomaly Detection")
        st.write("This section identifies unusual sales patterns and potential data anomalies.")
        
        if not filtered_df.empty:
            # Create tabs for different types of anomaly detection
            anomaly_tab1, anomaly_tab2 = st.tabs(["Sales Outliers", "Category Pattern Changes"])
            
            with anomaly_tab1:
                st.subheader("Sales Outliers Detection")
                st.write("Detecting unusual spikes or drops in sales data using statistical methods.")
                
                # Group data by month for analysis
                monthly_data = filtered_df.groupby(['Month', 'MonthSort']).agg({
                    'Net Sales': 'sum',
                    'Sold': 'sum'
                }).reset_index().sort_values('MonthSort')
                
                if len(monthly_data) >= 3:  # Need at least a few points for outlier detection
                    # Calculate Z-scores for sales
                    monthly_mean = monthly_data['Net Sales'].mean()
                    monthly_std = monthly_data['Net Sales'].std()
                    
                    if monthly_std > 0:  # Avoid division by zero
                        monthly_data['Z_Score'] = (monthly_data['Net Sales'] - monthly_mean) / monthly_std
                        
                        # Flag potential outliers
                        threshold = 1.5  # Z-score threshold for outliers
                        monthly_data['Is_Outlier'] = abs(monthly_data['Z_Score']) > threshold
                        monthly_data['Outlier_Type'] = 'Normal'
                        monthly_data.loc[monthly_data['Z_Score'] > threshold, 'Outlier_Type'] = 'Unusually High'
                        monthly_data.loc[monthly_data['Z_Score'] < -threshold, 'Outlier_Type'] = 'Unusually Low'
                        
                        # Create a visualization of outliers
                        fig = go.Figure()
                        
                        # Plot normal points
                        normal_data = monthly_data[~monthly_data['Is_Outlier']]
                        fig.add_trace(go.Scatter(
                            x=normal_data['Month'],
                            y=normal_data['Net Sales'],
                            mode='lines+markers',
                            name='Normal Data',
                            marker=dict(color='blue', size=10),
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Plot high outliers
                        high_outliers = monthly_data[(monthly_data['Is_Outlier']) & (monthly_data['Z_Score'] > 0)]
                        if not high_outliers.empty:
                            fig.add_trace(go.Scatter(
                                x=high_outliers['Month'],
                                y=high_outliers['Net Sales'],
                                mode='markers',
                                name='Unusually High',
                                marker=dict(color='green', size=15, symbol='circle'),
                                text=[f"Z-Score: {z:.2f}<br>Sales: ${sale:.2f}" for z, sale in zip(high_outliers['Z_Score'], high_outliers['Net Sales'])],
                                hovertemplate='%{text}',
                                hoverlabel=dict(bgcolor="green")
                            ))
                        
                        # Plot low outliers
                        low_outliers = monthly_data[(monthly_data['Is_Outlier']) & (monthly_data['Z_Score'] < 0)]
                        if not low_outliers.empty:
                            fig.add_trace(go.Scatter(
                                x=low_outliers['Month'],
                                y=low_outliers['Net Sales'],
                                mode='markers',
                                name='Unusually Low',
                                marker=dict(color='red', size=15, symbol='circle'),
                                text=[f"Z-Score: {z:.2f}<br>Sales: ${sale:.2f}" for z, sale in zip(low_outliers['Z_Score'], low_outliers['Net Sales'])],
                                hovertemplate='%{text}',
                                hoverlabel=dict(bgcolor="red")
                            ))
                        
                        # Get month order for proper display
                        month_order = get_proper_month_order(monthly_data['Month'].unique().tolist())
                        
                        # Update layout
                        fig.update_layout(
                            title="Sales Outlier Detection",
                            xaxis_title="",
                            yaxis_title="Net Sales ($)",
                            xaxis={'categoryorder': 'array', 'categoryarray': month_order},
                            xaxis_tickangle=-45,
                            hovermode="closest"
                        )
                        
                        # Add a horizontal line for the mean
                        fig.add_hline(y=monthly_mean, line_dash="dash", line_color="gray", 
                                    annotation_text=f"Mean: ${monthly_mean:.2f}", annotation_position="bottom right")
                        
                        # Add upper and lower bound reference lines
                        upper_bound = monthly_mean + threshold * monthly_std
                        lower_bound = monthly_mean - threshold * monthly_std
                        
                        fig.add_hline(y=upper_bound, line_dash="dot", line_color="green", 
                                    annotation_text="Upper Threshold", annotation_position="top right")
                        fig.add_hline(y=lower_bound, line_dash="dot", line_color="red", 
                                    annotation_text="Lower Threshold", annotation_position="bottom right")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display outlier information
                        outliers = monthly_data[monthly_data['Is_Outlier']]
                        if not outliers.empty:
                            st.write("### Detected Outliers")
                            st.dataframe(outliers[['Month', 'Net Sales', 'Outlier_Type', 'Z_Score']].sort_values('Month'))
                            
                            st.write("### Insights")
                            for _, row in outliers.iterrows():
                                if row['Z_Score'] > threshold:
                                    st.info(f"📈 **{row['Month']}** shows unusually high sales (${row['Net Sales']:.2f}), which is {row['Z_Score']:.2f} standard deviations above average. Consider analyzing what went well during this month.")
                                else:
                                    st.warning(f"📉 **{row['Month']}** shows unusually low sales (${row['Net Sales']:.2f}), which is {abs(row['Z_Score']):.2f} standard deviations below average. This may warrant investigation.")
                        else:
                            st.write("No outliers detected in the sales data.")
                    else:
                        st.write("Not enough variation in the data to detect outliers.")
                else:
                    st.write("Need at least 3 months of data to detect outliers.")
            
            with anomaly_tab2:
                st.subheader("Category Pattern Changes")
                st.write("Detecting unusual changes in category sales distribution.")
                
                # Analyze category distribution changes over time
                if len(months) >= 2:  # Need at least two months for comparison
                    # Calculate category sales percentages for each month
                    category_month_data = filtered_df.groupby(['Month', 'MonthSort', 'Category Name']).agg({
                        'Net Sales': 'sum'
                    }).reset_index()
                    
                    # Get total sales by month for percentage calculation
                    monthly_totals = filtered_df.groupby(['Month', 'MonthSort']).agg({
                        'Net Sales': 'sum'
                    }).reset_index()
                    
                    # Merge the data
                    category_data = pd.merge(category_month_data, monthly_totals, 
                                          on=['Month', 'MonthSort'], suffixes=('', '_Total'))
                    
                    # Calculate percentage
                    category_data['Percentage'] = (category_data['Net Sales'] / category_data['Net Sales_Total']) * 100
                    
                    # Sort by MonthSort for chronological order
                    category_data = category_data.sort_values('MonthSort')
                    
                    # Get top 5 categories by total sales
                    top_categories = filtered_df.groupby('Category Name')['Net Sales'].sum().sort_values(ascending=False).head(5).index.tolist()
                    
                    # Create a figure showing category share changes
                    fig = go.Figure()
                    
                    # Get month order for x-axis
                    month_order = get_proper_month_order(category_data['Month'].unique().tolist())
                    
                    # Create traces for each top category
                    for category in top_categories:
                        category_month_data = category_data[category_data['Category Name'] == category]
                        
                        if not category_month_data.empty:
                            fig.add_trace(go.Scatter(
                                x=category_month_data['Month'],
                                y=category_month_data['Percentage'],
                                mode='lines+markers',
                                name=category,
                                text=[f"{pct:.1f}%" for pct in category_month_data['Percentage']],
                                textposition="top center"
                            ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Category Share Trends",
                        xaxis_title="",
                        yaxis_title="% of Total Sales",
                        xaxis={'categoryorder': 'array', 'categoryarray': month_order},
                        xaxis_tickangle=-45,
                        yaxis_ticksuffix="%",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate significant category shifts
                    st.subheader("Significant Category Shifts")
                    st.write("Analyzing significant month-over-month changes in category market share.")
                    
                    # Calculate month-over-month percentage point changes for each category
                    category_shifts = []
                    
                    for category in filtered_df['Category Name'].unique():
                        cat_data = category_data[category_data['Category Name'] == category].sort_values('MonthSort')
                        
                        if len(cat_data) >= 2:  # Need at least two months for comparison
                            for i in range(1, len(cat_data)):
                                prev_month = cat_data.iloc[i-1]['Month']
                                curr_month = cat_data.iloc[i]['Month']
                                prev_pct = cat_data.iloc[i-1]['Percentage']
                                curr_pct = cat_data.iloc[i]['Percentage']
                                
                                # Calculate absolute and relative changes
                                abs_change = curr_pct - prev_pct
                                rel_change = (abs_change / prev_pct * 100) if prev_pct > 0 else float('inf')
                                
                                category_shifts.append({
                                    'Category': category,
                                    'Period': f"{prev_month} to {curr_month}",
                                    'Previous Share': prev_pct,
                                    'Current Share': curr_pct,
                                    'Absolute Change': abs_change,
                                    'Relative Change': rel_change
                                })
                    
                    # Convert to DataFrame
                    if category_shifts:
                        shifts_df = pd.DataFrame(category_shifts)
                        
                        # Filter for significant shifts
                        significant_shifts = shifts_df[
                            (abs(shifts_df['Absolute Change']) > 2) |  # More than 2 percentage points change
                            (abs(shifts_df['Relative Change']) > 20)   # More than 20% relative change
                        ].sort_values('Absolute Change', ascending=False)
                        
                        if not significant_shifts.empty:
                            # Create a separate DataFrame for display with formatted values
                            # This ensures sorting happens correctly before formatting
                            display_df = significant_shifts.copy()
                            
                            # Keep original numeric values for sorting and calculations
                            display_df['Previous Share Display'] = display_df['Previous Share'].apply(lambda x: f"{x:.2f}%")
                            display_df['Current Share Display'] = display_df['Current Share'].apply(lambda x: f"{x:.2f}%")
                            display_df['Absolute Change Display'] = display_df['Absolute Change'].apply(lambda x: f"{x:+.2f}%")
                            display_df['Relative Change Display'] = display_df['Relative Change'].apply(
                                lambda x: f"{x:+.2f}%" if not pd.isna(x) and not np.isinf(x) else "N/A"
                            )
                            
                            # Create display DataFrame with formatted columns
                            display_columns = ['Category', 'Period', 'Previous Share Display', 'Current Share Display', 
                                              'Absolute Change Display', 'Relative Change Display']
                            
                            # Show the formatted DataFrame
                            st.dataframe(display_df[display_columns].rename(columns={
                                'Previous Share Display': 'Previous Share',
                                'Current Share Display': 'Current Share',
                                'Absolute Change Display': 'Absolute Change',
                                'Relative Change Display': 'Relative Change'
                            }), use_container_width=True)
                            
                            # Highlight top most significant shifts
                            st.subheader("Key Insights")
                            top_increases = significant_shifts.nlargest(3, 'Absolute Change')
                            top_decreases = significant_shifts.nsmallest(3, 'Absolute Change')
                            
                            for _, row in top_increases.iterrows():
                                # Convert values to numeric if they're formatted strings
                                prev_share = row['Previous Share']
                                curr_share = row['Current Share']
                                abs_change = row['Absolute Change']
                                
                                if isinstance(prev_share, str):
                                    prev_share = float(prev_share.strip('%'))
                                if isinstance(curr_share, str):
                                    curr_share = float(curr_share.strip('%'))
                                if isinstance(abs_change, str):
                                    abs_change = float(abs_change.strip('%').strip('+'))
                                
                                st.success(f"📈 **{row['Category']}** increased significantly from {prev_share:.2f}% to {curr_share:.2f}% ({abs_change:+.2f} points) during {row['Period']}.")
                                
                            for _, row in top_decreases.iterrows():
                                # Convert values to numeric if they're formatted strings
                                prev_share = row['Previous Share']
                                curr_share = row['Current Share']
                                abs_change = row['Absolute Change']
                                
                                if isinstance(prev_share, str):
                                    prev_share = float(prev_share.strip('%'))
                                if isinstance(curr_share, str):
                                    curr_share = float(curr_share.strip('%'))
                                if isinstance(abs_change, str):
                                    abs_change = float(abs_change.strip('%').strip('+'))
                                
                                st.warning(f"📉 **{row['Category']}** decreased significantly from {prev_share:.2f}% to {curr_share:.2f}% ({abs_change:+.2f} points) during {row['Period']}.")
                        else:
                            st.write("No significant category shifts detected in the selected time period.")
                    else:
                        st.write("Insufficient data to analyze category shifts.")
                else:
                    st.write("Need at least two months of data to analyze category pattern changes.")
                 
        with tab5:
            st.subheader("Market Basket Analysis")
            st.write("This section analyzes which product categories may be commonly purchased together.")
            
            if not filtered_df.empty:
                # Category mix and correlation analysis
                # First, get total sales by category by month
                cat_month_sales = filtered_df.groupby(['Month', 'Category Name'])['Net Sales'].sum().reset_index()
                
                # Pivot to get categories as columns
                cat_pivot = cat_month_sales.pivot(index='Month', columns='Category Name', values='Net Sales').fillna(0)
                
                # Calculate correlation matrix
                corr_matrix = cat_pivot.corr()
                
                # Create a heatmap of the correlation matrix
                fig = px.imshow(
                    corr_matrix,
                    title="Category Sales Correlation Matrix",
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',  # Red-Blue scale, reversed so blue=positive
                    zmin=-1,  # Set min value
                    zmax=1     # Set max value
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Interpretation**: Higher positive values (blue) indicate categories that tend to perform similarly - they may be commonly purchased together or by the same customer segments.")
                
                # Find top category correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if i != j:  # Skip self-correlations
                            cat1 = corr_matrix.columns[i]
                            cat2 = corr_matrix.columns[j]
                            corr_value = corr_matrix.iloc[i, j]
                            corr_pairs.append({'Category 1': cat1, 'Category 2': cat2, 'Correlation': corr_value})
                
                # Create DataFrame of correlation pairs
                corr_df = pd.DataFrame(corr_pairs)
                
                # Show top positive and negative correlations
                if len(corr_df) > 0:
                    st.subheader("Top Category Relationships")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Strongest Positive Correlations**")
                        top_positive = corr_df.nlargest(5, 'Correlation')
                        positive_shown = False
                        for _, row in top_positive.iterrows():
                            if row['Correlation'] > 0.5:  # Only show strong correlations
                                st.info(f"**{row['Category 1']}** and **{row['Category 2']}** tend to sell together (correlation: {row['Correlation']:.2f})")
                                positive_shown = True
                        if not positive_shown and not top_positive.empty:
                            st.write("No strong positive correlations found.")
                    
                    with col2:
                        st.write("**Strongest Negative Correlations**")
                        top_negative = corr_df.nsmallest(5, 'Correlation')
                        negative_shown = False
                        for _, row in top_negative.iterrows():
                            if row['Correlation'] < -0.5:  # Only show strong negative correlations
                                st.warning(f"**{row['Category 1']}** tends to sell opposite to **{row['Category 2']}** (correlation: {row['Correlation']:.2f})")
                                negative_shown = True
                        if not negative_shown and not top_negative.empty:
                            st.write("No strong negative correlations found.")
                
                # Category share over time
                st.subheader("Category Share Evolution")
                
                # Calculate total sales by month
                monthly_totals = filtered_df.groupby('Month')['Net Sales'].sum().reset_index()
                monthly_totals = monthly_totals.rename(columns={'Net Sales': 'Total Sales'})
                
                # Merge with category sales and calculate percentage
                cat_share = pd.merge(cat_month_sales, monthly_totals, on='Month')
                cat_share['Percentage'] = (cat_share['Net Sales'] / cat_share['Total Sales']) * 100
                
                # Get month order for proper chronological display
                month_order = get_proper_month_order(filtered_df['Month'].unique().tolist())
                
                # Create stacked area chart
                fig = px.area(
                    cat_share,
                    x='Month',
                    y='Percentage',
                    color='Category Name',
                    title="Category Share Evolution Over Time",
                    labels={
                        'Month': '',
                        'Percentage': '% of Total Sales',
                        'Category Name': 'Category'
                    }
                )
                
                # Force the x-axis to respect the month order
                fig.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': month_order},
                    xaxis_tickangle=-45,
                    yaxis_ticksuffix='%'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.write("**Insight**: This chart shows how each category's share of total sales changes over time, helping identify which categories are growing or declining in importance.")
                
                # Add product bundling suggestions based on correlated categories
                st.subheader("Product Bundling Suggestions")
                st.write("These suggestions are based on product categories that tend to sell together:")
                
                # Get top 3 correlation pairs for bundling (if correlation > 0.4)
                bundle_candidates = corr_df[corr_df['Correlation'] > 0.4].nlargest(3, 'Correlation')
                
                if not bundle_candidates.empty:
                    for _, row in bundle_candidates.iterrows():
                        # Get top products from each category
                        cat1_products = filtered_df[filtered_df['Category Name'] == row['Category 1']].groupby('Name')['Net Sales'].sum().nlargest(2)
                        cat2_products = filtered_df[filtered_df['Category Name'] == row['Category 2']].groupby('Name')['Net Sales'].sum().nlargest(2)
                        
                        if not cat1_products.empty and not cat2_products.empty:
                            cat1_product_names = cat1_products.index.tolist()
                            cat2_product_names = cat2_products.index.tolist()
                            
                            st.success(f"**Bundle Opportunity**: Products from {row['Category 1']} and {row['Category 2']} (correlation: {row['Correlation']:.2f})")
                            st.write(f"Top products to bundle:")
                            st.write(f"• From {row['Category 1']}: {', '.join(cat1_product_names)}")
                            st.write(f"• From {row['Category 2']}: {', '.join(cat2_product_names)}")
                else:
                    st.info("No strong product bundling opportunities identified based on current data. Consider collecting more sales data to improve correlation analysis.")
            else:
                st.write("No data available for the selected period.")

#---------------------------
# BUSINESS INSIGHTS PAGE
#---------------------------
elif page == "Business Insights":
    st.title("Business Insights Dashboard")
    st.subheader(f"Analysis Period: {date_range}")
    
    if not filtered_df.empty:
        # Calculate key metrics
        total_revenue = filtered_df['Net Sales'].sum()
        total_units = filtered_df['Sold'].sum()
        total_products = filtered_df['Name'].nunique()
        avg_price = total_revenue / total_units if total_units > 0 else 0
        
        # Display KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Total Units Sold", f"{total_units:,}")
        col3.metric("Unique Products", f"{total_products}")
        col4.metric("Average Unit Price", f"${avg_price:.2f}")
        
        # Create tabs for different insights
        tab1, tab2 = st.tabs(["Revenue Insights", "Product Insights"])
        
        with tab1:
            st.subheader("Revenue Driver Analysis")
            
            # Revenue by Category
            category_rev = filtered_df.groupby('Category Name')['Net Sales'].sum().reset_index()
            category_rev['Percentage'] = (category_rev['Net Sales'] / total_revenue) * 100
            category_rev = category_rev.sort_values('Net Sales', ascending=False)
            
            # Pareto Analysis (80/20 rule)
            category_rev['Cumulative Percentage'] = category_rev['Percentage'].cumsum()
            
            # Create a dual-axis chart for Pareto analysis
            fig = go.Figure()
            
            # Add bars for category revenue with improved labels
            fig.add_trace(go.Bar(
                x=category_rev['Category Name'],
                y=category_rev['Net Sales'],
                name='Revenue',
                text=category_rev['Net Sales'],
                texttemplate='$%{text:.0f}',
                textposition='outside',
                marker_color='royalblue'
            ))
            
            # Add line for cumulative percentage with improved labels
            fig.add_trace(go.Scatter(
                x=category_rev['Category Name'],
                y=category_rev['Cumulative Percentage'],
                name='Cumulative %',
                text=category_rev['Cumulative Percentage'],
                texttemplate='%{text:.1f}%',
                textposition='top center',
                mode='lines+markers+text',
                marker_color='red',
                yaxis='y2'
            ))
            
            # Add 80% reference line
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=80,
                x1=len(category_rev)-0.5,
                y1=80,
                yref='y2',
                line=dict(color="green", width=2, dash="dash")
            )
            
            # Update layout with dual y-axes
            fig.update_layout(
                title="Pareto Analysis: Revenue by Category",
                xaxis_title="Category",
                yaxis_title="Revenue ($)",
                yaxis2=dict(
                    title=dict(
                        text="Cumulative Percentage",
                        font=dict(color="red")
                    ),
                    tickfont=dict(color="red"),
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                    ticksuffix="%"
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Insight box
            threshold_categories = category_rev[category_rev['Cumulative Percentage'] <= 80]['Category Name'].tolist()
            st.info(f"**Pareto Insight**: {len(threshold_categories)} out of {len(category_rev)} categories ({len(threshold_categories)/len(category_rev)*100:.1f}%) generate 80% of your revenue. Focus on these key categories: {', '.join(threshold_categories)}")
            
            # Revenue growth by month
            if len(months) > 1:
                st.subheader("Revenue Growth Analysis")
                
                # Calculate monthly growth
                monthly_rev = filtered_df.groupby(['Month', 'MonthSort'])['Net Sales'].sum().reset_index()
                monthly_rev = monthly_rev.sort_values('MonthSort')
                monthly_rev['Previous'] = monthly_rev['Net Sales'].shift(1)
                monthly_rev['Growth'] = (monthly_rev['Net Sales'] - monthly_rev['Previous']) / monthly_rev['Previous'] * 100
                monthly_rev['GrowthLabel'] = monthly_rev['Growth'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")
                
                # Create growth chart
                month_order = get_proper_month_order(monthly_rev['Month'].unique().tolist())
                
                fig = go.Figure()
                
                # Add revenue bars with improved labels
                fig.add_trace(go.Bar(
                    x=monthly_rev['Month'],
                    y=monthly_rev['Net Sales'],
                    name="Revenue",
                    text=monthly_rev['Net Sales'],
                    texttemplate='$%{text:.0f}',
                    textposition='outside',
                    marker_color="royalblue"
                ))
                
                # Add growth line
                fig.add_trace(go.Scatter(
                    x=monthly_rev['Month'],
                    y=monthly_rev['Growth'],
                    name="Growth %",
                    mode="lines+markers+text",
                    text=monthly_rev['GrowthLabel'],
                    textposition="top center",
                    line=dict(color="green"),
                    marker=dict(color="green"),
                    yaxis="y2"
                ))
                
                # Update layout
                fig.update_layout(
                    title="Monthly Revenue with Growth Rate",
                    xaxis=dict(
                        title="",
                        categoryorder='array',
                        categoryarray=month_order,
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title="Revenue ($)"
                    ),
                    yaxis2=dict(
                        title=dict(
                            text="Growth Rate (%)",
                            font=dict(color="green")
                        ),
                        tickfont=dict(color="green"),
                        overlaying="y",
                        side="right",
                        ticksuffix="%"
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth insights
                avg_growth = monthly_rev['Growth'].mean()
                positive_months = (monthly_rev['Growth'] > 0).sum()
                total_months_with_growth = len(monthly_rev) - 1  # Exclude first month which has no growth data
                
                if total_months_with_growth > 0:
                    positive_growth_ratio = positive_months / total_months_with_growth
                    
                    # Show insights based on growth patterns
                    if avg_growth > 5:
                        st.success(f"**Growth Insight**: Your business is growing at a healthy average rate of {avg_growth:.1f}% per month with {positive_months} out of {total_months_with_growth} months showing positive growth.")
                    elif avg_growth > 0:
                        st.info(f"**Growth Insight**: Your business is showing modest growth averaging {avg_growth:.1f}% per month with {positive_months} out of {total_months_with_growth} months showing positive growth.")
                    else:
                        st.warning(f"**Growth Insight**: Your business is experiencing decline with an average monthly growth rate of {avg_growth:.1f}%. Only {positive_months} out of {total_months_with_growth} months showed positive growth.")
        
        with tab2:
            st.subheader("Product Performance Insights")
            
            # Calculate product metrics
            product_metrics = filtered_df.groupby('Name').agg({
                'Net Sales': 'sum',
                'Sold': 'sum',
                'Category Name': 'first'
            }).reset_index()
            
            product_metrics['Avg Price'] = product_metrics['Net Sales'] / product_metrics['Sold']
            product_metrics['Revenue Share'] = (product_metrics['Net Sales'] / total_revenue) * 100
            
            # Top and bottom products
            top_products = product_metrics.nlargest(10, 'Net Sales')
            bottom_products = product_metrics[product_metrics['Sold'] > 0].nsmallest(10, 'Net Sales')
            
            # Create breakdown visualization
            st.subheader("Revenue Concentration")
            
            # Calculate product count thresholds
            product_count = len(product_metrics)
            top_10_pct_count = max(int(product_count * 0.1), 1)
            top_20_pct_count = max(int(product_count * 0.2), 1)
            top_50_pct_count = max(int(product_count * 0.5), 1)
            
            # Calculate revenue for each segment
            top_10_pct_rev = product_metrics.nlargest(top_10_pct_count, 'Net Sales')['Net Sales'].sum()
            top_20_pct_rev = product_metrics.nlargest(top_20_pct_count, 'Net Sales')['Net Sales'].sum()
            top_50_pct_rev = product_metrics.nlargest(top_50_pct_count, 'Net Sales')['Net Sales'].sum()
            
            # Calculate percentages
            top_10_pct = (top_10_pct_rev / total_revenue) * 100
            top_20_pct = (top_20_pct_rev / total_revenue) * 100
            top_50_pct = (top_50_pct_rev / total_revenue) * 100
            
            # Create visualization
            concentration_data = pd.DataFrame([
                {'Segment': f'Top 10% ({top_10_pct_count} products)', 'Revenue Share': top_10_pct},
                {'Segment': f'Top 20% ({top_20_pct_count} products)', 'Revenue Share': top_20_pct},
                {'Segment': f'Top 50% ({top_50_pct_count} products)', 'Revenue Share': top_50_pct},
                {'Segment': 'All Products', 'Revenue Share': 100}
            ])
            
            fig = px.bar(
                concentration_data,
                x='Segment',
                y='Revenue Share',
                title="Revenue Concentration Analysis",
                labels={'Revenue Share': 'Percentage of Total Revenue (%)', 'Segment': ''},
                text='Revenue Share'  # Use the actual values instead of auto formatting
            )
            
            # Format text to show the percentage sign
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode='show')
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide insight
            if top_10_pct > 50:
                st.warning(f"**Concentration Risk**: Your top 10% of products ({top_10_pct_count} products) generate {top_10_pct:.1f}% of your revenue. This high concentration creates risk if sales decline for these few products.")
            else:
                st.info(f"**Diversification Insight**: Your top 10% of products ({top_10_pct_count} products) generate {top_10_pct:.1f}% of your revenue, showing a reasonably diversified product mix.")
            
            # Product performance metrics
            st.subheader("Product Performance Distribution")
            
            # Extract data for visualization
            product_count = len(product_metrics)
            zero_sales = product_metrics[product_metrics['Sold'] == 0].shape[0]
            low_sales = product_metrics[(product_metrics['Sold'] > 0) & (product_metrics['Sold'] < 5)].shape[0]
            medium_sales = product_metrics[(product_metrics['Sold'] >= 5) & (product_metrics['Sold'] < 20)].shape[0]
            high_sales = product_metrics[product_metrics['Sold'] >= 20].shape[0]
            
            # Calculate percentages
            zero_pct = (zero_sales / product_count) * 100
            low_pct = (low_sales / product_count) * 100
            medium_pct = (medium_sales / product_count) * 100
            high_pct = (high_sales / product_count) * 100
            
            # Create distribution visualization with custom values
            custom_values = [zero_sales, low_sales, medium_sales, high_sales]
            custom_labels = ['No Sales', '1-4 Units', '5-19 Units', '20+ Units']
            custom_percentages = [zero_pct, low_pct, medium_pct, high_pct]
            
            distribution_data = pd.DataFrame({
                'Category': custom_labels,
                'Count': custom_values,
                'Percentage': custom_percentages
            })
            
            fig = px.pie(
                distribution_data,
                values='Count',
                names='Category',
                title="Product Performance Distribution",
                hole=0.4,
                custom_data=['Percentage']
            )
            
            # Ensure labels show both category and percentage
            fig.update_traces(
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.1f}%'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide insights
            if zero_sales > 0:
                st.warning(f"**Dead Stock Alert**: {zero_sales} products ({zero_pct:.1f}% of your catalog) had zero sales during this period. Consider discounting or discontinuing these items.")
            
            # Price point analysis
            st.subheader("Price Point Effectiveness")
            
            # Create price bins
            price_bins = [0, 10, 20, 30, 50, 100, 1000]
            bin_labels = ['$0-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100+']  
            
            product_metrics['Price Range'] = pd.cut(
                product_metrics['Avg Price'],
                bins=price_bins,
                labels=bin_labels,
                right=False
            )
            
            # Get performance by price range
            price_performance = product_metrics.groupby('Price Range').agg({
                'Net Sales': 'sum',
                'Sold': 'sum',
                'Name': 'count'
            }).reset_index()
            
            price_performance = price_performance.rename(columns={'Name': 'Product Count'})
            price_performance['Revenue Share'] = (price_performance['Net Sales'] / total_revenue) * 100
            price_performance['Unit Share'] = (price_performance['Sold'] / total_units) * 100
            
            # Visualization
            fig = go.Figure()
            
            # Add revenue bars with improved labels
            fig.add_trace(go.Bar(
                x=price_performance['Price Range'],
                y=price_performance['Revenue Share'],
                name='% of Revenue',
                text=price_performance['Revenue Share'],
                texttemplate='%{text:.1f}%',
                textposition='outside',
                marker_color='royalblue'
            ))
            
            # Add unit share bars with improved labels
            fig.add_trace(go.Bar(
                x=price_performance['Price Range'],
                y=price_performance['Unit Share'],
                name='% of Units',
                text=price_performance['Unit Share'],
                texttemplate='%{text:.1f}%',
                textposition='outside',
                marker_color='lightgreen'
            ))
            
            # Update layout
            fig.update_layout(
                title='Price Range Analysis: Revenue vs Unit Share',
                xaxis_title='Price Range',
                yaxis_title='Percentage (%)',
                barmode='group',
                yaxis_ticksuffix='%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find most effective price range
            best_revenue_range = price_performance.loc[price_performance['Revenue Share'].idxmax(), 'Price Range']
            best_unit_range = price_performance.loc[price_performance['Unit Share'].idxmax(), 'Price Range']
            
            if best_revenue_range == best_unit_range:
                st.success(f"**Pricing Sweet Spot**: Your most effective price range is {best_revenue_range}, which dominates both in revenue ({price_performance['Revenue Share'].max():.1f}%) and unit sales ({price_performance['Unit Share'].max():.1f}%).")
            else:
                st.info(f"**Pricing Insight**: Your highest revenue share comes from the {best_revenue_range} price range ({price_performance.loc[price_performance['Revenue Share'].idxmax(), 'Revenue Share']:.1f}%), while most units are sold in the {best_unit_range} range ({price_performance.loc[price_performance['Unit Share'].idxmax(), 'Unit Share']:.1f}%).")
            
            # Product gaps analysis
            low_performing_categories = filtered_df.groupby('Category Name')['Net Sales'].sum().nsmallest(3).reset_index()
            low_performing_categories['Category Name'] = low_performing_categories['Category Name'].astype(str)
            
            if not low_performing_categories.empty:
                st.subheader("Opportunity Analysis")
                st.write("These categories might represent opportunities for new products or marketing focus:")
                
                for _, row in low_performing_categories.iterrows():
                    category_product_count = filtered_df[filtered_df['Category Name'] == row['Category Name']]['Name'].nunique()
                    st.info(f"**{row['Category Name']}**: ${row['Net Sales']:.2f} in revenue with {category_product_count} products. This may represent an opportunity for additional product offerings or focused marketing.")
    else:
        st.write("No data available for the selected period.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard visualizes vape store monthly sales data.
    Data is read from CSV files in the monthly_sales directory.
    """
)

# Main function to run the app
if __name__ == "__main__":
    # Display a notice at the bottom of the page
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.8em;">
        Developed for Madvapes Sales Analysis | Last updated: March 2025
        </div>
        """, 
        unsafe_allow_html=True
    )