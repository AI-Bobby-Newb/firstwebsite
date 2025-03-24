from flask import Flask, render_template, jsonify
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from scipy import stats
from monthly_sales_analyzer import MonthlySalesAnalyzer


app = Flask(__name__)

# Initialize analyzer
analyzer = MonthlySalesAnalyzer("monthly_sales")

# Get reports data
def get_reports(): #function
    reports = []
    for report_file in analyzer.get_available_reports():
        report = analyzer.parse_report(report_file)
        if report:
            reports.append(report)
    return sorted(reports, key=lambda x: (x['year'], {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                                     'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                                     'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}.get(x['month'], 0)))
reports = get_reports()
# Create combined dataframe with all sales data
def get_combined_data(reports): #function
    all_dfs = []
    month_order = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                   'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    for report in reports:
        df = report['data'].copy()
        df = df[df['Category Name'] != 'TOTAL']
        df['Month'] = f"{report['month']} {report['year']}"
        df['MonthSort'] = f"{report['year']}-{month_order.get(report['month'], '00')}"
        df['MonthIndex'] = int(month_order.get(report['month'], '00'))
        df['Year'] = int(report['year'])
        all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs)
    return pd.DataFrame()
combined_df = get_combined_data(reports)
# Calculate data quality score
def get_data_quality_score(df, reports): #function
    score = 100
    warnings = []
    if df.empty or len(reports) == 0:
        return 0, ["No data available"]
    current_date = datetime.now()
    month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                   'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
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
    if most_recent:
        current_month = current_date.month
        current_year = current_date.year
        months_diff = (current_year - most_recent_year) * 12 + (current_month - most_recent_month)
        if months_diff > 3:
            score -= min(40, months_diff * 10)
            warnings.append(f"Data is {months_diff} months old")
        elif months_diff > 1:
            score -= months_diff * 5
            warnings.append(f"Data is {months_diff} months old")
    else:
        score -= 40
        warnings.append("Cannot determine data recency")
    if len(reports) >= 2:
        def get_sort_key(report):
            if isinstance(report, dict) and 'month' in report and 'year' in report:
                month = report['month']
                year = int(report['year'])
                month_num = month_order.get(month, 0)
                return (year, month_num)
            return (0, 0)
        sorted_reports = sorted(reports, key=get_sort_key)
        gaps = 0
        for i in range(1, len(sorted_reports)):
            prev_report = sorted_reports[i - 1]
            curr_report = sorted_reports[i]
            if isinstance(prev_report, dict) and isinstance(curr_report, dict):
                prev_year = int(prev_report['year'])
                prev_month = month_order.get(prev_report['month'], 0)
                curr_year = int(curr_report['year'])
                curr_month = month_order.get(curr_report['month'], 0)
                months_diff = (curr_year - prev_year) * 12 + (curr_month - prev_month)
                if months_diff > 1:
                    gaps += months_diff - 1
        if gaps > 0:
            score -= min(20, gaps * 5)
            warnings.append(f"{gaps} missing month(s) in data")
    if not df.empty:
        monthly_sales = df.groupby('Month')['Net Sales'].sum().reset_index()
        if len(monthly_sales) >= 3:
            mean_sales = monthly_sales['Net Sales'].mean()
            std_sales = monthly_sales['Net Sales'].std()
            cv = std_sales / mean_sales if mean_sales > 0 else 0
            if cv > 0.5:
                score -= min(20, int(cv * 20))
                warnings.append("High sales variability detected")
    score = max(0, min(100, score))
    return score, warnings
quality_score, quality_warnings = get_data_quality_score(combined_df, reports)
# Custom function to get appropriate month-year ordering
def get_proper_month_order(month_data): #function
    month_mapping = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    ordered_months = []
    for month in month_data:
        parts = month.split()
        if len(parts) >= 2:
            month_name = parts[0]
            year = parts[1]
            month_num = month_mapping.get(month_name[:3], '00')
            ordered_months.append((year, month_num, month))
    ordered_months.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in ordered_months]
# Get and sort months chronologically
months = combined_df['Month'].unique().tolist()
months = get_proper_month_order(months)
# Add a historical timeline display
def group_months_by_year(months_list): # function
    years_dict = {}
    for month in months_list:
        parts = month.split()
        month_name = parts[0]
        year = parts[1]
        if year not in years_dict:
            years_dict[year] = []
        years_dict[year].append(month_name)
    return years_dict
years_dict = group_months_by_year(months)

@app.route('/')
def index():
    """Dashboard Overview Page"""
    # Prepare data (same as in your Streamlit app)
    total_sales = combined_df['Net Sales'].sum()
    total_units = combined_df['Sold'].sum()
    avg_price = total_sales / total_units if total_units > 0 else 0
    unique_products = combined_df['Name'].nunique()
     # Get and sort months chronologically
    months = combined_df['Month'].unique().tolist()
    month_order = get_proper_month_order(months)
    monthly_sales = combined_df.groupby(['Month', 'MonthSort'])['Net Sales'].sum().reset_index()
    monthly_sales = monthly_sales.sort_values('MonthSort')
    category_sales = combined_df.groupby('Category Name')['Net Sales'].sum().reset_index()
    category_sales = category_sales.sort_values('Net Sales', ascending=False).head(5)
    top_products = combined_df.groupby('Name')['Net Sales'].sum().reset_index()
    top_products = top_products.sort_values('Net Sales', ascending=False).head(10)
    top_units = combined_df.groupby('Name')['Sold'].sum().reset_index()
    top_units = top_units.sort_values('Sold', ascending=False).head(10)
    
    # Pass data to the template
    return render_template('index.html', #Create this html
                           total_sales=total_sales,
                           total_units=total_units,
                           avg_price=avg_price,
                           unique_products=unique_products,
                           monthly_sales=monthly_sales.to_dict(orient='records'),
                           category_sales=category_sales.to_dict(orient='records'),
                           top_products=top_products.to_dict(orient='records'),
                           top_units=top_units.to_dict(orient='records'),
                           month_order=month_order,
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

@app.route('/monthly_analysis')
def monthly_analysis():
    """Monthly Analysis Page"""
    monthly_data = combined_df.groupby(['Month', 'MonthSort', 'Year', 'MonthIndex']).agg({
        'Net Sales': 'sum',
        'Sold': 'sum'
    }).reset_index()
    monthly_data = monthly_data.sort_values('MonthSort')
    if len(monthly_data) > 1:
        monthly_data['Sales Growth'] = monthly_data['Net Sales'].pct_change() * 100
        monthly_data['Units Growth'] = monthly_data['Sold'].pct_change() * 100
    
     # Get and sort months chronologically
    months = combined_df['Month'].unique().tolist()
    month_order = get_proper_month_order(months)
    return render_template('monthly_analysis.html',
                           monthly_data=monthly_data.to_dict(orient='records'),
                           month_order=month_order,
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

@app.route('/product_analysis')
def product_analysis():
    """Product Analysis Page"""
    categories = sorted(combined_df['Category Name'].unique().tolist())
    
    product_df = combined_df
   
    product_perf = product_df.groupby('Name').agg({
        'Net Sales': 'sum',
        'Sold': 'sum',
        'Category Name': 'first'
    }).reset_index()
    product_perf['Avg Price'] = product_perf['Net Sales'] / product_perf['Sold']
    product_perf = product_perf.sort_values('Net Sales', ascending=False)
    top_products = product_perf.head(10)
    
    return render_template('product_analysis.html',
                           categories=categories,
                           top_products=top_products.to_dict(orient='records'),
                           product_perf=product_perf.to_dict(orient='records'),
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

@app.route('/category_analysis')
def category_analysis():
    """Category Analysis Page"""
    category_data = combined_df.groupby('Category Name').agg({
        'Net Sales': 'sum',
        'Sold': 'sum',
        'Name': pd.Series.nunique
    }).reset_index()
    total_sales = category_data['Net Sales'].sum()
    category_data['Percent of Sales'] = (category_data['Net Sales'] / total_sales * 100)
    category_data['Avg Price'] = category_data['Net Sales'] / category_data['Sold']
    category_data = category_data.sort_values('Net Sales', ascending=False)
    category_data = category_data.rename(columns={'Name': 'Products'})
    top_categories = category_data.head(5)['Category Name'].tolist()
    
    # Get and sort months chronologically
    months = combined_df['Month'].unique().tolist()
    month_order = get_proper_month_order(months)
    
    return render_template('category_analysis.html',
                           category_data=category_data.to_dict(orient='records'),
                           top_categories=top_categories,
                           month_order = month_order,
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

@app.route('/top_products')
def top_products():
    """Top Products Page"""
    product_data = combined_df.groupby(['Name', 'Category Name']).agg({
        'Net Sales': 'sum',
        'Sold': 'sum'
    }).reset_index()
    product_data['Avg Price'] = product_data['Net Sales'] / product_data['Sold']
    
    return render_template('top_products.html',
                           product_data=product_data.to_dict(orient='records'),
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

@app.route('/advanced_analytics')
def advanced_analytics():
    """Advanced Analytics Page"""
    # Get and sort months chronologically
    months = combined_df['Month'].unique().tolist()
    month_order = get_proper_month_order(months)

    return render_template('advanced_analytics.html',
                           combined_df=combined_df,
                           month_order = month_order,
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

@app.route('/business_insights')
def business_insights():
    """Business Insights"""
    total_revenue = combined_df['Net Sales'].sum()
    total_units = combined_df['Sold'].sum()
    total_products = combined_df['Name'].nunique()
    avg_price = total_revenue / total_units if total_units > 0 else 0
    
    product_metrics = combined_df.groupby('Name').agg({
        'Net Sales': 'sum',
        'Sold': 'sum',
        'Category Name': 'first'
    }).reset_index()

    product_metrics['Avg Price'] = product_metrics['Net Sales'] / product_metrics['Sold']
    product_metrics['Revenue Share'] = (product_metrics['Net Sales'] / total_revenue) * 100
    
    # Get and sort months chronologically
    months = combined_df['Month'].unique().tolist()
    month_order = get_proper_month_order(months)
    
    return render_template('business_insights.html',
                           total_revenue = total_revenue,
                           total_units = total_units,
                           total_products = total_products,
                           avg_price = avg_price,
                           product_metrics = product_metrics.to_dict(orient='records'),
                           month_order = month_order,
                           years_dict = years_dict,
                           quality_score = quality_score,
                           quality_warnings = quality_warnings,
                           months=months)

if __name__ == '__main__':
    app.run(debug=True)