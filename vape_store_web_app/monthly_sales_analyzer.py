import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

class MonthlySalesAnalyzer:
    """Analyzes monthly sales data from CSV files"""
    
    def __init__(self, sales_dir):
        """Initialize with the directory containing sales report CSV files"""
        self.sales_dir = sales_dir
        
        # Ensure output directory exists
        self.output_dir = "reports"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_available_reports(self):
        """Get a list of available monthly sales reports"""
        reports = []
        
        if os.path.exists(self.sales_dir):
            for filename in os.listdir(self.sales_dir):
                if filename.endswith('.csv'):
                    reports.append(filename)
        
        # Define month order for chronological sorting
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        # Custom sort function for report files based on month and year
        def sort_reports(report_file):
            try:
                # Handle variations in file naming
                if 'Items Sales Report' in report_file:
                    parts = report_file.replace('Items Sales Report ', '').split()
                elif 'Item Sales Report' in report_file:
                    parts = report_file.replace('Item Sales Report ', '').split()
                elif 'tem Sales Report' in report_file:  # Handle the typo in July file
                    parts = report_file.replace('tem Sales Report ', '').split()
                else:
                    # If none of the expected patterns match, just return the filename
                    return report_file
                
                # Ensure we have at least 2 parts in the array and proper month, year extraction
                if len(parts) < 2:
                    return (0, 0)  # Invalid format, sort to beginning
                    
                month = parts[0]
                year = int(parts[1])
                month_num = month_order.get(month[:3], 0)
                
                # Print debug info for troubleshooting
                print(f"Sorting file: {report_file}, Month: {month}, Year: {year}, Month Num: {month_num}")
                
                # Return a tuple for sorting: (year, month_num)
                return (year, month_num)
            except Exception:
                # If parsing fails, return a default value
                return (0, 0)
        
        # Sort reports by year and month
        return sorted(reports, key=sort_reports)
    
    def parse_report(self, report_file):
        """Parse a sales report and return a dictionary with the data"""
        filepath = os.path.join(self.sales_dir, report_file)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            # Parse the file name to extract month and year
            # Create a normalized filename by handling all variations
            if 'Items Sales Report' in report_file:
                parts = report_file.replace('Items Sales Report ', '').split()
            elif 'Item Sales Report' in report_file:
                parts = report_file.replace('Item Sales Report ', '').split()
            elif 'tem Sales Report' in report_file:  # Handle the typo in July file
                parts = report_file.replace('tem Sales Report ', '').split()
            else:
                # If we can't parse it, try a more generic approach
                parts = report_file.split()
                if len(parts) < 2:
                    raise Exception(f"Unable to parse filename: {report_file}")
            
            # Get month and year from parts
            month = parts[0]
            year = parts[1]
            
            # Read the CSV file with pandas
            df = pd.read_csv(filepath, skiprows=2)
            
            # Convert sales columns to numeric
            df['Gross Sales'] = df['Gross Sales'].str.replace('$', '').str.replace(',', '').astype(float)
            df['Net Sales'] = df['Net Sales'].str.replace('$', '').str.replace(',', '').astype(float)
            
            # Convert 'Sold' to numeric if it isn't already
            if df['Sold'].dtype == 'object':
                df['Sold'] = df['Sold'].astype(int)
            
            # Read the first two rows to get the title and period
            with open(filepath, 'r', encoding='utf-8') as file:
                first_two_lines = [next(file) for _ in range(2)]
            
            report_title = first_two_lines[0].split(',')[0].strip('"')
            report_period = first_two_lines[1].split(',')[0].strip('"')
            
            # Return a dictionary with report info and data
            return {
                'title': report_title,
                'period': report_period,
                'month': month,
                'year': year,
                'filename': report_file,
                'data': df
            }
        except Exception as e:
            print(f"Error parsing report {report_file}: {e}")
            return None
    
    def list_available_reports(self):
        """Display a list of available sales reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAvailable Monthly Sales Reports:")
        print("-" * 60)
        
        for i, report_file in enumerate(reports, 1):
            report = self.parse_report(report_file)
            if report:
                print(f"{i}. {report['month']} {report['year']} - {report_file}")
            else:
                print(f"{i}. {report_file} (Error: Unable to parse)")
        
        input("\nPress Enter to continue...")
    
    def view_report_details(self):
        """View details of a specific sales report"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAvailable Monthly Sales Reports:")
        print("-" * 60)
        
        for i, report_file in enumerate(reports, 1):
            report = self.parse_report(report_file)
            if report:
                print(f"{i}. {report['month']} {report['year']} - {report_file}")
            else:
                print(f"{i}. {report_file} (Error: Unable to parse)")
        
        try:
            choice = int(input("\nSelect a report number to view (0 to cancel): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(reports):
                self.display_report_details(reports[choice-1])
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    def display_report_details(self, report_file):
        """Display detailed information about a specific report"""
        report = self.parse_report(report_file)
        
        if not report:
            print(f"\nError: Could not parse report {report_file}")
            input("\nPress Enter to continue...")
            return
        
        # Get the data and filter out the "TOTAL" category to avoid double counting
        df = report['data']
        df_filtered = df[df['Category Name'] != 'TOTAL']
        
        # Calculate summary statistics
        total_items_sold = df_filtered['Sold'].sum()
        total_gross_sales = df_filtered['Gross Sales'].sum()
        total_net_sales = df_filtered['Net Sales'].sum()
        
        # Calculate sales by category
        category_sales = df.groupby('Category Name').agg({
            'Sold': 'sum',
            'Net Sales': 'sum'
        }).sort_values('Net Sales', ascending=False)
        
        # Get top 10 products by units sold
        top_by_units = df.sort_values('Sold', ascending=False).head(10)
        
        # Get top 10 products by revenue
        top_by_revenue = df.sort_values('Net Sales', ascending=False).head(10)
        
        # Display the report
        print(f"\n{report['title']} - {report['period']}")
        print("-" * 60)
        print(f"Total Items Sold: {total_items_sold}")
        print(f"Total Gross Sales: ${total_gross_sales:.2f}")
        print(f"Total Net Sales: ${total_net_sales:.2f}")
        
        # Display sales by category
        print("\nSales by Category:")
        print(f"{'Category':<20}| {'Units Sold':<10}| {'Net Sales':<12}| {'% of Total':<10}")
        print("-" * 60)
        
        for category, row in category_sales.iterrows():
            category_name = category[:18] if len(category) > 18 else category
            percent = (row['Net Sales'] / total_net_sales) * 100
            print(f"{category_name:<20}| {row['Sold']:<10}| ${row['Net Sales']:<10.2f}| {percent:<9.2f}%")
        
        # Display top items by revenue
        print("\nTop 10 Products by Revenue:")
        print(f"{'#':<3}| {'Name':<40}| {'Category':<15}| {'Units':<6}| {'Sales':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(top_by_revenue.iterrows(), 1):
            # Handle potential NaN values in Name and Category
            name_val = str(row['Name']) if not pd.isna(row['Name']) else "Unknown"
            category_val = str(row['Category Name']) if not pd.isna(row['Category Name']) else "Unknown"
            
            name = name_val[:38] if len(name_val) > 38 else name_val
            category = category_val[:13] if len(category_val) > 13 else category_val
            print(f"{i:<3}| {name:<40}| {category:<15}| {row['Sold']:<6}| ${row['Net Sales']:<8.2f}")
        
        # Display top items by units sold
        print("\nTop 10 Products by Units Sold:")
        print(f"{'#':<3}| {'Name':<40}| {'Category':<15}| {'Units':<6}| {'Sales':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(top_by_units.iterrows(), 1):
            # Handle potential NaN values in Name and Category
            name_val = str(row['Name']) if not pd.isna(row['Name']) else "Unknown"
            category_val = str(row['Category Name']) if not pd.isna(row['Category Name']) else "Unknown"
            
            name = name_val[:38] if len(name_val) > 38 else name_val
            category = category_val[:13] if len(category_val) > 13 else category_val
            print(f"{i:<3}| {name:<40}| {category:<15}| {row['Sold']:<6}| ${row['Net Sales']:<8.2f}")
        
        # Show options for viewing all items or returning
        print("\n1. View All Items")
        print("2. Export Summary to CSV")
        print("0. Back")
        
        choice = input("\nSelect an option: ")
        
        if choice == '1':
            self.view_all_items(report)
        elif choice == '2':
            self.export_report_summary(report)
    
    def view_all_items(self, report):
        """Display all items in a report with pagination"""
        df = report['data']
        items_per_page = 20
        total_pages = (len(df) + items_per_page - 1) // items_per_page
        current_page = 1
        
        while True:
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(df))
            
            print(f"\nAll Items in {report['title']} (Page {current_page}/{total_pages}):")
            print(f"{'#':<4}| {'Name':<40}| {'Category':<15}| {'Units':<6}| {'Sales':<10}")
            print("-" * 80)
            
            for i, (_, row) in enumerate(df.iloc[start_idx:end_idx].iterrows(), start_idx + 1):
                # Handle potential NaN values
                name_val = str(row['Name']) if not pd.isna(row['Name']) else "Unknown"
                category_val = str(row['Category Name']) if not pd.isna(row['Category Name']) else "Unknown"
                
                name = name_val[:38] if len(name_val) > 38 else name_val
                category = category_val[:13] if len(category_val) > 13 else category_val
                print(f"{i:<4}| {name:<40}| {category:<15}| {row['Sold']:<6}| ${row['Net Sales']:<8.2f}")
            
            print(f"\nPage {current_page}/{total_pages}")
            print("N: Next Page | P: Previous Page | 0: Back")
            
            choice = input("\nSelect an option: ").lower()
            
            if choice == 'n' and current_page < total_pages:
                current_page += 1
            elif choice == 'p' and current_page > 1:
                current_page -= 1
            elif choice == '0':
                break
            else:
                print("Invalid option.")
    
    def compare_reports(self):
        """Compare multiple sales reports"""
        reports = self.get_available_reports()
        
        if len(reports) < 2:
            print("\nAt least 2 reports are needed for comparison.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAvailable Reports for Comparison:")
        for i, report_file in enumerate(reports, 1):
            report = self.parse_report(report_file)
            if report:
                print(f"{i}. {report['month']} {report['year']} - {report_file}")
        
        # Get reports to compare
        selections = input("\nSelect reports to compare (comma-separated numbers): ")
        try:
            selected_indices = [int(x.strip()) - 1 for x in selections.split(',')]
            
            # Validate selections
            if any(idx < 0 or idx >= len(reports) for idx in selected_indices):
                print("Invalid selection.")
                return
            
            if len(selected_indices) < 2:
                print("Please select at least 2 reports to compare.")
                return
            
            selected_reports = [self.parse_report(reports[idx]) for idx in selected_indices]
            if any(report is None for report in selected_reports):
                print("Error parsing one or more of the selected reports.")
                return
            
            self.display_report_comparison(selected_reports)
            
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers.")
    
    def display_report_comparison(self, reports):
        """Display a comparison between multiple reports"""
        # Sort reports by date (assuming month names can be used for sorting)
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        def get_sort_key(report):
            month = report['month']
            year = int(report['year'])
            month_num = month_order.get(month[:3], 0)
            return (year, month_num)
        
        sorted_reports = sorted(reports, key=get_sort_key)
        
        # Calculate period labels and totals
        periods = [f"{report['month']} {report['year']}" for report in sorted_reports]
        totals = []
        
        for report in sorted_reports:
            df = report['data']
            total_sold = df['Sold'].sum()
            total_sales = df['Net Sales'].sum()
            totals.append((total_sold, total_sales))
        
        # Display comparison summary
        print("\nSales Comparison Summary:")
        print("-" * 80)
        
        # Display header
        print(f"{'Metric':<20}|" + "".join(f" {period:<15}|" for period in periods))
        print("-" * 80)
        
        # Display total items sold
        print(f"{'Total Items Sold':<20}|" + "".join(f" {total[0]:<15}|" for total in totals))
        
        # Display total revenue
        print(f"{'Total Revenue':<20}|" + "".join(f" ${total[1]:<14.2f}|" for total in totals))
        
        # Calculate and display growth rates between periods
        if len(sorted_reports) >= 2:
            print("\nMonth-over-Month Growth:")
            print("-" * 80)
            print(f"{'Metric':<20}|" + "".join(f" {periods[i+1]:<15}|" for i in range(len(periods)-1)))
            print("-" * 80)
            
            # Units sold growth
            units_growth = []
            for i in range(len(totals)-1):
                if totals[i][0] > 0:  # Avoid division by zero
                    growth = ((totals[i+1][0] - totals[i][0]) / totals[i][0]) * 100
                    units_growth.append(f"{growth:.2f}%")
                else:
                    units_growth.append("N/A")
            
            print(f"{'Units Sold Growth':<20}|" + "".join(f" {growth:<15}|" for growth in units_growth))
            
            # Revenue growth
            revenue_growth = []
            for i in range(len(totals)-1):
                if totals[i][1] > 0:  # Avoid division by zero
                    growth = ((totals[i+1][1] - totals[i][1]) / totals[i][1]) * 100
                    revenue_growth.append(f"{growth:.2f}%")
                else:
                    revenue_growth.append("N/A")
            
            print(f"{'Revenue Growth':<20}|" + "".join(f" {growth:<15}|" for growth in revenue_growth))
        
        # Category comparison
        print("\nCategory Comparison:")
        print("-" * 80)
        
        # Get all unique categories across reports
        all_categories = set()
        for report in sorted_reports:
            categories = report['data']['Category Name'].unique()
            all_categories.update(categories)
        
        # For each category, get sales across reports
        for category in sorted(all_categories):
            category_sales = []
            for report in sorted_reports:
                df = report['data']
                cat_data = df[df['Category Name'] == category]
                cat_sales = cat_data['Net Sales'].sum() if not cat_data.empty else 0
                category_sales.append(cat_sales)
            
            # Display the category sales
            print(f"{category[:19]:<20}|" + "".join(f" ${sales:<14.2f}|" for sales in category_sales))
        
        # Top product comparison (find products that appear in multiple reports)
        print("\nTop Products Across Reports:")
        print("-" * 80)
        
        # Combine all product data
        all_products = {}
        for i, report in enumerate(sorted_reports):
            df = report['data']
            for _, row in df.iterrows():
                name = row['Name']
                if name not in all_products:
                    all_products[name] = [0] * len(sorted_reports)
                all_products[name][i] = row['Net Sales']
        
        # Calculate total sales for each product across all reports
        product_totals = {name: sum(sales) for name, sales in all_products.items()}
        
        # Get top 10 products by total sales
        top_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Display header
        print(f"{'Product':<40}|" + "".join(f" {period:<15}|" for period in periods))
        print("-" * (40 + 17 * len(periods)))
        
        # Display sales for each top product
        for name, _ in top_products:
            name_display = name[:38] if len(name) > 38 else name
            sales = all_products[name]
            print(f"{name_display:<40}|" + "".join(f" ${sales[i]:<14.2f}|" for i in range(len(sales))))
        
        # Option to export
        export_choice = input("\nExport comparison to CSV? (y/n): ")
        if export_choice.lower() == 'y':
            self.export_comparison(sorted_reports, all_categories, all_products, top_products)
        
        input("\nPress Enter to continue...")
    
    def export_comparison(self, reports, all_categories, all_products, top_products):
        """Export comparison data to a CSV file"""
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            periods = [f"{report['month']}{report['year']}" for report in reports]
            period_str = "_vs_".join(periods)
            filename = f"sales_comparison_{period_str}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                periods_full = [f"{report['month']} {report['year']}" for report in reports]
                writer.writerow(["Sales Comparison Report", ", ".join(periods_full)])
                writer.writerow([])
                
                # Write summary data
                writer.writerow(["Summary"])
                writer.writerow(["Metric"] + periods_full)
                
                totals_sold = [report['data']['Sold'].sum() for report in reports]
                totals_sales = [report['data']['Net Sales'].sum() for report in reports]
                
                writer.writerow(["Total Items Sold"] + totals_sold)
                writer.writerow(["Total Revenue"] + [f"${sales:.2f}" for sales in totals_sales])
                
                # Write growth data if applicable
                if len(reports) >= 2:
                    writer.writerow([])
                    writer.writerow(["Growth Rates"])
                    growth_periods = [f"{periods_full[i]} to {periods_full[i+1]}" for i in range(len(periods_full)-1)]
                    writer.writerow(["Period"] + growth_periods)
                    
                    # Units sold growth
                    units_growth = []
                    for i in range(len(totals_sold)-1):
                        if totals_sold[i] > 0:
                            growth = ((totals_sold[i+1] - totals_sold[i]) / totals_sold[i]) * 100
                            units_growth.append(f"{growth:.2f}%")
                        else:
                            units_growth.append("N/A")
                    
                    writer.writerow(["Units Sold Growth"] + units_growth)
                    
                    # Revenue growth
                    revenue_growth = []
                    for i in range(len(totals_sales)-1):
                        if totals_sales[i] > 0:
                            growth = ((totals_sales[i+1] - totals_sales[i]) / totals_sales[i]) * 100
                            revenue_growth.append(f"{growth:.2f}%")
                        else:
                            revenue_growth.append("N/A")
                    
                    writer.writerow(["Revenue Growth"] + revenue_growth)
                
                # Write category comparison
                writer.writerow([])
                writer.writerow(["Category Comparison"])
                writer.writerow(["Category"] + periods_full)
                
                for category in sorted(all_categories):
                    category_sales = []
                    for report in reports:
                        df = report['data']
                        cat_data = df[df['Category Name'] == category]
                        cat_sales = cat_data['Net Sales'].sum() if not cat_data.empty else 0
                        category_sales.append(f"${cat_sales:.2f}")
                    
                    writer.writerow([category] + category_sales)
                
                # Write top products comparison
                writer.writerow([])
                writer.writerow(["Top Products Comparison"])
                writer.writerow(["Product"] + periods_full + ["Total Sales"])
                
                for name, total in top_products:
                    sales = all_products[name]
                    row_data = [name]
                    for i in range(len(sales)):
                        row_data.append(f"${sales[i]:.2f}")
                    row_data.append(f"${total:.2f}")
                    writer.writerow(row_data)
            
            print(f"\nComparison exported to {filepath}")
            return True
        except Exception as e:
            print(f"\nError exporting comparison: {e}")
            return False
    
    def search_products(self):
        """Search for specific products across reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        search_term = input("\nEnter product name or category to search for: ").lower()
        
        if not search_term:
            print("Search term cannot be empty.")
            input("\nPress Enter to continue...")
            return
        
        print(f"\nSearching for '{search_term}' across {len(reports)} reports...")
        
        results = []
        
        for report_file in reports:
            report = self.parse_report(report_file)
            
            if report:
                df = report['data']
                
                # Search in product names and categories
                matches = df[
                    df['Name'].str.lower().str.contains(search_term) | 
                    df['Category Name'].str.lower().str.contains(search_term)
                ]
                
                if not matches.empty:
                    # Add report info to each match
                    matches['Report'] = f"{report['month']} {report['year']}"
                    results.append(matches)
        
        if not results:
            print(f"No matches found for '{search_term}'.")
            input("\nPress Enter to continue...")
            return
        
        # Combine all results
        all_results = pd.concat(results)
        
        # Display results
        print(f"\nFound {len(all_results)} matches for '{search_term}':")
        print(f"{'Report':<15}| {'Name':<40}| {'Category':<15}| {'Units':<6}| {'Sales':<10}")
        print("-" * 90)
        
        for _, row in all_results.iterrows():
            name = row['Name'][:38] if len(row['Name']) > 38 else row['Name']
            category = row['Category Name'][:13] if len(row['Category Name']) > 13 else row['Category Name']
            print(f"{row['Report']:<15}| {name:<40}| {category:<15}| {row['Sold']:<6}| ${row['Net Sales']:<8.2f}")
        
        # Option to export
        export_choice = input("\nExport search results to CSV? (y/n): ")
        if export_choice.lower() == 'y':
            self.export_search_results(search_term, all_results)
        
        input("\nPress Enter to continue...")
    
    def export_search_results(self, search_term, results):
        """Export search results to CSV file"""
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"search_results_{search_term.replace(' ', '_')}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Export to CSV
            results.to_csv(filepath, index=False)
            
            print(f"\nSearch results exported to {filepath}")
            return True
        except Exception as e:
            print(f"\nError exporting search results: {e}")
            return False
    
    def export_summary(self):
        """Export summary of all reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nGenerating summary of all sales reports...")
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if not parsed_reports:
            print("Error parsing reports.")
            input("\nPress Enter to continue...")
            return
        
        # Sort reports by date
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        def get_sort_key(report):
            month = report['month']
            year = int(report['year'])
            month_num = month_order.get(month[:3], 0)
            return (year, month_num)
        
        sorted_reports = sorted(parsed_reports, key=get_sort_key)
        
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"sales_summary_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(["Monthly Sales Summary Report", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Write monthly totals
                writer.writerow(["Monthly Totals"])
                writer.writerow(["Month", "Items Sold", "Gross Sales", "Net Sales"])
                
                for report in sorted_reports:
                    df = report['data']
                    total_sold = df['Sold'].sum()
                    total_gross = df['Gross Sales'].sum()
                    total_net = df['Net Sales'].sum()
                    
                    writer.writerow([
                        f"{report['month']} {report['year']}",
                        total_sold,
                        f"${total_gross:.2f}",
                        f"${total_net:.2f}"
                    ])
                
                # Calculate overall totals
                overall_sold = sum(report['data']['Sold'].sum() for report in sorted_reports)
                overall_gross = sum(report['data']['Gross Sales'].sum() for report in sorted_reports)
                overall_net = sum(report['data']['Net Sales'].sum() for report in sorted_reports)
                
                writer.writerow([])
                writer.writerow(["Overall Totals", overall_sold, f"${overall_gross:.2f}", f"${overall_net:.2f}"])
                
                # Get all unique categories
                all_categories = set()
                for report in sorted_reports:
                    categories = report['data']['Category Name'].unique()
                    all_categories.update(categories)
                
                # Write category summary
                writer.writerow([])
                writer.writerow(["Category Summary"])
                writer.writerow(["Category", "Total Items Sold", "Total Net Sales", "% of Total Sales"])
                
                category_totals = {}
                for category in all_categories:
                    total_sold = 0
                    total_sales = 0
                    
                    for report in sorted_reports:
                        df = report['data']
                        cat_data = df[df['Category Name'] == category]
                        total_sold += cat_data['Sold'].sum()
                        total_sales += cat_data['Net Sales'].sum()
                    
                    category_totals[category] = (total_sold, total_sales)
                
                # Write category data sorted by sales
                for category, (sold, sales) in sorted(category_totals.items(), key=lambda x: x[1][1], reverse=True):
                    percent = (sales / overall_net) * 100 if overall_net > 0 else 0
                    writer.writerow([category, sold, f"${sales:.2f}", f"{percent:.2f}%"])
                
                # Get top products across all reports
                all_products = {}
                for report in sorted_reports:
                    df = report['data']
                    for _, row in df.iterrows():
                        name = row['Name']
                        if name not in all_products:
                            all_products[name] = {'sold': 0, 'sales': 0, 'category': row['Category Name']}
                        
                        all_products[name]['sold'] += row['Sold']
                        all_products[name]['sales'] += row['Net Sales']
                
                # Write top products by revenue
                writer.writerow([])
                writer.writerow(["Top 20 Products by Revenue"])
                writer.writerow(["Rank", "Product", "Category", "Total Units Sold", "Total Net Sales"])
                
                top_by_revenue = sorted(all_products.items(), key=lambda x: x[1]['sales'], reverse=True)[:20]
                for i, (name, data) in enumerate(top_by_revenue, 1):
                    writer.writerow([i, name, data['category'], data['sold'], f"${data['sales']:.2f}"])
                
                # Write top products by units sold
                writer.writerow([])
                writer.writerow(["Top 20 Products by Units Sold"])
                writer.writerow(["Rank", "Product", "Category", "Total Units Sold", "Total Net Sales"])
                
                top_by_units = sorted(all_products.items(), key=lambda x: x[1]['sold'], reverse=True)[:20]
                for i, (name, data) in enumerate(top_by_units, 1):
                    writer.writerow([i, name, data['category'], data['sold'], f"${data['sales']:.2f}"])
            
            print(f"\nSummary exported to {filepath}")
        except Exception as e:
            print(f"\nError exporting summary: {e}")
        
        input("\nPress Enter to continue...")
    
    def export_report_summary(self, report):
        """Export a summary of a single report to CSV"""
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            period = f"{report['month']}_{report['year']}"
            filename = f"report_summary_{period}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Get the data
            df = report['data']
            total_items_sold = df['Sold'].sum()
            total_gross_sales = df['Gross Sales'].sum()
            total_net_sales = df['Net Sales'].sum()
            
            # Calculate sales by category
            category_sales = df.groupby('Category Name').agg({
                'Sold': 'sum',
                'Net Sales': 'sum'
            }).sort_values('Net Sales', ascending=False)
            
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header info
                writer.writerow([report['title']])
                writer.writerow([report['period']])
                writer.writerow([])
                
                # Write summary
                writer.writerow(["Summary"])
                writer.writerow(["Total Items Sold", total_items_sold])
                writer.writerow(["Total Gross Sales", f"${total_gross_sales:.2f}"])
                writer.writerow(["Total Net Sales", f"${total_net_sales:.2f}"])
                writer.writerow([])
                
                # Write category sales
                writer.writerow(["Sales by Category"])
                writer.writerow(["Category", "Units Sold", "Net Sales", "% of Total"])
                
                for category, row in category_sales.iterrows():
                    percent = (row['Net Sales'] / total_net_sales) * 100
                    writer.writerow([
                        category,
                        row['Sold'],
                        f"${row['Net Sales']:.2f}",
                        f"{percent:.2f}%"
                    ])
                
                writer.writerow([])
                
                # Write top items by revenue
                writer.writerow(["Top 20 Items by Revenue"])
                writer.writerow(["Rank", "Name", "Category", "Units Sold", "Net Sales"])
                
                top_by_revenue = df.sort_values('Net Sales', ascending=False).head(20)
                for i, (_, row) in enumerate(top_by_revenue.iterrows(), 1):
                    writer.writerow([
                        i,
                        row['Name'],
                        row['Category Name'],
                        row['Sold'],
                        f"${row['Net Sales']:.2f}"
                    ])
                
                # Write all items
                writer.writerow([])
                writer.writerow(["Complete Item List"])
                writer.writerow(["Name", "Category", "Units Sold", "Gross Sales", "Net Sales"])
                
                for _, row in df.iterrows():
                    writer.writerow([
                        row['Name'],
                        row['Category Name'],
                        row['Sold'],
                        f"${row['Gross Sales']:.2f}",
                        f"${row['Net Sales']:.2f}"
                    ])
            
            print(f"\nReport summary exported to {filepath}")
            return True
        except Exception as e:
            print(f"\nError exporting report: {e}")
            return False
            
    def export_monthly_report(self):
        """Export a monthly report summary to CSV"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAvailable Monthly Reports:")
        for i, report_file in enumerate(reports, 1):
            report = self.parse_report(report_file)
            if report:
                print(f"{i}. {report['month']} {report['year']} - {report_file}")
        
        try:
            choice = int(input("\nSelect a report to export (0 to cancel): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(reports):
                report = self.parse_report(reports[choice-1])
                if report:
                    self.export_report_summary(report)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        
        input("\nPress Enter to continue...")
    
    def export_category_summary(self):
        """Export a summary of sales by category"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nGenerating category summary...")
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if not parsed_reports:
            print("Error parsing reports.")
            input("\nPress Enter to continue...")
            return
        
        # Get all unique categories
        all_categories = set()
        for report in parsed_reports:
            categories = report['data']['Category Name'].unique()
            all_categories.update(categories)
        
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"category_summary_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(["Category Sales Summary", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Calculate overall totals
                overall_net = sum(report['data']['Net Sales'].sum() for report in parsed_reports)
                
                # Write category summary
                writer.writerow(["Category Summary"])
                writer.writerow(["Category", "Total Units Sold", "Total Net Sales", "% of Total Sales"])
                
                category_totals = {}
                for category in all_categories:
                    total_sold = 0
                    total_sales = 0
                    
                    for report in parsed_reports:
                        df = report['data']
                        cat_data = df[df['Category Name'] == category]
                        total_sold += cat_data['Sold'].sum()
                        total_sales += cat_data['Net Sales'].sum()
                    
                    category_totals[category] = (total_sold, total_sales)
                
                # Write category data sorted by sales
                for category, (sold, sales) in sorted(category_totals.items(), key=lambda x: x[1][1], reverse=True):
                    percent = (sales / overall_net) * 100 if overall_net > 0 else 0
                    writer.writerow([category, sold, f"${sales:.2f}", f"{percent:.2f}%"])
                
                # For each category, show monthly breakdown
                writer.writerow([])
                writer.writerow(["Monthly Category Breakdown"])
                
                # Sort reports by date
                month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                
                def get_sort_key(report):
                    month = report['month']
                    year = int(report['year'])
                    month_num = month_order.get(month[:3], 0)
                    return (year, month_num)
                
                sorted_reports = sorted(parsed_reports, key=get_sort_key)
                periods = [f"{report['month']} {report['year']}" for report in sorted_reports]
                
                # Write breakdown for each category
                for category in sorted(category_totals.items(), key=lambda x: x[1][1], reverse=True):
                    category_name = category[0]
                    writer.writerow([])
                    writer.writerow([f"Category: {category_name}"])
                    writer.writerow(["Month", "Units Sold", "Net Sales", "% of Month Total"])
                    
                    for report in sorted_reports:
                        df = report['data']
                        month_total = df['Net Sales'].sum()
                        cat_data = df[df['Category Name'] == category_name]
                        cat_sold = cat_data['Sold'].sum()
                        cat_sales = cat_data['Net Sales'].sum()
                        cat_pct = (cat_sales / month_total) * 100 if month_total > 0 else 0
                        
                        writer.writerow([
                            f"{report['month']} {report['year']}",
                            cat_sold,
                            f"${cat_sales:.2f}",
                            f"{cat_pct:.2f}%"
                        ])
            
            print(f"\nCategory summary exported to {filepath}")
        except Exception as e:
            print(f"\nError exporting summary: {e}")
        
        input("\nPress Enter to continue...")
    
    def export_product_rankings(self):
        """Export product rankings across all reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nExporting product rankings...")
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if not parsed_reports:
            print("Error parsing reports.")
            input("\nPress Enter to continue...")
            return
        
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"product_rankings_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Combine all product data across reports
            all_products = {}
            
            for report in parsed_reports:
                df = report['data']
                for _, row in df.iterrows():
                    name = row['Name']
                    if name not in all_products:
                        all_products[name] = {
                            'category': row['Category Name'],
                            'sold': 0,
                            'sales': 0,
                        }
                    
                    all_products[name]['sold'] += row['Sold']
                    all_products[name]['sales'] += row['Net Sales']
            
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(["Product Rankings", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Write top products by revenue
                writer.writerow(["Top Products by Revenue"])
                writer.writerow(["Rank", "Product", "Category", "Total Units Sold", "Total Net Sales", "Average Price"])
                
                top_by_revenue = sorted(all_products.items(), key=lambda x: x[1]['sales'], reverse=True)
                for i, (name, data) in enumerate(top_by_revenue, 1):
                    avg_price = data['sales'] / data['sold'] if data['sold'] > 0 else 0
                    writer.writerow([
                        i,
                        name,
                        data['category'],
                        data['sold'],
                        f"${data['sales']:.2f}",
                        f"${avg_price:.2f}"
                    ])
                
                # Write top products by units sold
                writer.writerow([])
                writer.writerow(["Top Products by Units Sold"])
                writer.writerow(["Rank", "Product", "Category", "Total Units Sold", "Total Net Sales", "Average Price"])
                
                top_by_units = sorted(all_products.items(), key=lambda x: x[1]['sold'], reverse=True)
                for i, (name, data) in enumerate(top_by_units, 1):
                    avg_price = data['sales'] / data['sold'] if data['sold'] > 0 else 0
                    writer.writerow([
                        i,
                        name,
                        data['category'],
                        data['sold'],
                        f"${data['sales']:.2f}",
                        f"${avg_price:.2f}"
                    ])
                
                # Write efficiency metrics (sales per unit)
                writer.writerow([])
                writer.writerow(["Highest Revenue per Unit (min 5 units sold)"])
                writer.writerow(["Rank", "Product", "Category", "Units Sold", "Net Sales", "Revenue per Unit"])
                
                high_efficiency = sorted(
                    [(name, data) for name, data in all_products.items() if data['sold'] >= 5],
                    key=lambda x: x[1]['sales'] / x[1]['sold'] if x[1]['sold'] > 0 else 0,
                    reverse=True
                )
                
                for i, (name, data) in enumerate(high_efficiency[:50], 1):
                    revenue_per_unit = data['sales'] / data['sold'] if data['sold'] > 0 else 0
                    writer.writerow([
                        i,
                        name,
                        data['category'],
                        data['sold'],
                        f"${data['sales']:.2f}",
                        f"${revenue_per_unit:.2f}"
                    ])
            
            print(f"\nProduct rankings exported to {filepath}")
        except Exception as e:
            print(f"\nError exporting rankings: {e}")
        
        input("\nPress Enter to continue...")
    
    def export_custom_report(self):
        """Export a custom report based on user criteria"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nCustom Report Generator")
        print("-" * 60)
        
        # Choose reports to include
        print("\nAvailable Reports:")
        for i, report_file in enumerate(reports, 1):
            report = self.parse_report(report_file)
            if report:
                print(f"{i}. {report['month']} {report['year']} - {report_file}")
        
        selections = input("\nSelect reports to include (comma-separated numbers, or 'all'): ")
        
        if selections.lower() == 'all':
            selected_reports = [self.parse_report(rf) for rf in reports]
            selected_reports = [r for r in selected_reports if r]
        else:
            try:
                selected_indices = [int(x.strip()) - 1 for x in selections.split(',')]
                selected_reports = [self.parse_report(reports[idx]) for idx in selected_indices if 0 <= idx < len(reports)]
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers or 'all'.")
                input("\nPress Enter to continue...")
                return
        
        if not selected_reports:
            print("No valid reports selected.")
            input("\nPress Enter to continue...")
            return
        
        # Choose report content
        print("\nSelect Report Content:")
        print("1. Sales Summary")
        print("2. Category Analysis")
        print("3. Top Products")
        print("4. Bottom Products")
        print("5. All of the Above")
        
        content_choice = input("\nEnter your choice (1-5): ")
        
        if content_choice not in ['1', '2', '3', '4', '5']:
            print("Invalid choice.")
            input("\nPress Enter to continue...")
            return
        
        # Generate the report
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            report_type = {
                '1': 'summary',
                '2': 'category',
                '3': 'top_products',
                '4': 'bottom_products',
                '5': 'full'
            }.get(content_choice, 'custom')
            
            filename = f"custom_report_{report_type}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                included_months = ", ".join([f"{r['month']} {r['year']}" for r in selected_reports])
                writer.writerow([f"Custom Sales Report - {included_months}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Combine data from all selected reports
                all_data = pd.concat([report['data'] for report in selected_reports])
                
                # Sales Summary
                if content_choice in ['1', '5']:
                    writer.writerow(["Sales Summary"])
                    total_items = all_data['Sold'].sum()
                    total_gross = all_data['Gross Sales'].sum()
                    total_net = all_data['Net Sales'].sum()
                    
                    writer.writerow(["Total Items Sold", total_items])
                    writer.writerow(["Total Gross Sales", f"${total_gross:.2f}"])
                    writer.writerow(["Total Net Sales", f"${total_net:.2f}"])
                    writer.writerow([])
                    
                    # Monthly breakdown
                    writer.writerow(["Monthly Breakdown"])
                    writer.writerow(["Month", "Items Sold", "Gross Sales", "Net Sales"])
                    
                    for report in selected_reports:
                        df = report['data']
                        month_items = df['Sold'].sum()
                        month_gross = df['Gross Sales'].sum()
                        month_net = df['Net Sales'].sum()
                        
                        writer.writerow([
                            f"{report['month']} {report['year']}",
                            month_items,
                            f"${month_gross:.2f}",
                            f"${month_net:.2f}"
                        ])
                    
                    writer.writerow([])
                
                # Category Analysis
                if content_choice in ['2', '5']:
                    writer.writerow(["Category Analysis"])
                    category_data = all_data.groupby('Category Name').agg({
                        'Sold': 'sum',
                        'Net Sales': 'sum'
                    }).sort_values('Net Sales', ascending=False)
                    
                    writer.writerow(["Category", "Items Sold", "Net Sales", "% of Total"])
                    
                    for category, row in category_data.iterrows():
                        pct = (row['Net Sales'] / total_net) * 100 if total_net > 0 else 0
                        writer.writerow([
                            category,
                            row['Sold'],
                            f"${row['Net Sales']:.2f}",
                            f"{pct:.2f}%"
                        ])
                    
                    writer.writerow([])
                
                # Top Products
                if content_choice in ['3', '5']:
                    writer.writerow(["Top 50 Products by Revenue"])
                    
                    top_products = all_data.groupby(['Name', 'Category Name']).agg({
                        'Sold': 'sum',
                        'Net Sales': 'sum'
                    }).sort_values('Net Sales', ascending=False).reset_index().head(50)
                    
                    writer.writerow(["Rank", "Product", "Category", "Units Sold", "Net Sales", "Avg Price"])
                    
                    for i, row in enumerate(top_products.iterrows(), 1):
                        _, data = row
                        avg_price = data['Net Sales'] / data['Sold'] if data['Sold'] > 0 else 0
                        writer.writerow([
                            i,
                            data['Name'],
                            data['Category Name'],
                            data['Sold'],
                            f"${data['Net Sales']:.2f}",
                            f"${avg_price:.2f}"
                        ])
                    
                    writer.writerow([])
                
                # Bottom Products
                if content_choice in ['4', '5']:
                    writer.writerow(["Bottom 50 Products by Revenue (with sales > 0)"])
                    
                    bottom_products = all_data.groupby(['Name', 'Category Name']).agg({
                        'Sold': 'sum',
                        'Net Sales': 'sum'
                    })
                    
                    # Filter out products with no sales
                    bottom_products = bottom_products[bottom_products['Net Sales'] > 0]
                    bottom_products = bottom_products.sort_values('Net Sales').reset_index().head(50)
                    
                    writer.writerow(["Rank", "Product", "Category", "Units Sold", "Net Sales", "Avg Price"])
                    
                    for i, row in enumerate(bottom_products.iterrows(), 1):
                        _, data = row
                        avg_price = data['Net Sales'] / data['Sold'] if data['Sold'] > 0 else 0
                        writer.writerow([
                            i,
                            data['Name'],
                            data['Category Name'],
                            data['Sold'],
                            f"${data['Net Sales']:.2f}",
                            f"${avg_price:.2f}"
                        ])
            
            print(f"\nCustom report exported to {filepath}")
        except Exception as e:
            print(f"\nError generating custom report: {e}")
        
        input("\nPress Enter to continue...")
            
    # Product Analysis Functions
    def top_selling_products(self):
        """Display top selling products by revenue and units across all reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if not parsed_reports:
            print("Error parsing reports.")
            input("\nPress Enter to continue...")
            return
        
        # Combine all data
        all_data = pd.concat([report['data'] for report in parsed_reports])
        
        # Group by product and calculate totals
        product_sales = all_data.groupby(['Name', 'Category Name']).agg({
            'Sold': 'sum',
            'Net Sales': 'sum'
        }).reset_index()
        
        # Calculate average price per unit
        product_sales['Avg Price'] = product_sales['Net Sales'] / product_sales['Sold']
        
        # Sort by revenue
        top_by_revenue = product_sales.sort_values('Net Sales', ascending=False).head(20)
        
        # Sort by units sold
        top_by_units = product_sales.sort_values('Sold', ascending=False).head(20)
        
        # Display top products by revenue
        print("\nTOP 20 PRODUCTS BY REVENUE:")
        print("=" * 80)
        
        revenue_data = []
        for i, (_, row) in enumerate(top_by_revenue.iterrows(), 1):
            revenue_data.append([
                i, 
                row['Name'][:40], 
                row['Category Name'][:15], 
                row['Sold'], 
                f"${row['Net Sales']:.2f}", 
                f"${row['Avg Price']:.2f}"
            ])
        
        print(tabulate(revenue_data, headers=[
            "Rank", "Product", "Category", "Units Sold", "Net Sales", "Avg Price"
        ], tablefmt="grid"))
        
        # Display top products by units sold
        print("\nTOP 20 PRODUCTS BY UNITS SOLD:")
        print("=" * 80)
        
        units_data = []
        for i, (_, row) in enumerate(top_by_units.iterrows(), 1):
            units_data.append([
                i, 
                row['Name'][:40], 
                row['Category Name'][:15], 
                row['Sold'], 
                f"${row['Net Sales']:.2f}", 
                f"${row['Avg Price']:.2f}"
            ])
        
        print(tabulate(units_data, headers=[
            "Rank", "Product", "Category", "Units Sold", "Net Sales", "Avg Price"
        ], tablefmt="grid"))
        
        export_choice = input("\nExport these rankings to CSV? (y/n): ")
        if export_choice.lower() == 'y':
            self.export_product_rankings()
        else:
            input("\nPress Enter to continue...")
    
    def worst_selling_products(self):
        """Display worst selling products by revenue and units across all reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if not parsed_reports:
            print("Error parsing reports.")
            input("\nPress Enter to continue...")
            return
        
        # Combine all data
        all_data = pd.concat([report['data'] for report in parsed_reports])
        
        # Group by product and calculate totals
        product_sales = all_data.groupby(['Name', 'Category Name']).agg({
            'Sold': 'sum',
            'Net Sales': 'sum'
        }).reset_index()
        
        # Filter out products with no sales
        product_sales = product_sales[product_sales['Net Sales'] > 0]
        
        # Calculate average price per unit
        product_sales['Avg Price'] = product_sales['Net Sales'] / product_sales['Sold']
        
        # Sort by revenue (ascending)
        bottom_by_revenue = product_sales.sort_values('Net Sales').head(20)
        
        # Sort by units sold (ascending)
        bottom_by_units = product_sales.sort_values('Sold').head(20)
        
        # Display bottom products by revenue
        print("\nBOTTOM 20 PRODUCTS BY REVENUE (with sales > 0):")
        print("=" * 80)
        
        revenue_data = []
        for i, (_, row) in enumerate(bottom_by_revenue.iterrows(), 1):
            revenue_data.append([
                i, 
                row['Name'][:40], 
                row['Category Name'][:15], 
                row['Sold'], 
                f"${row['Net Sales']:.2f}", 
                f"${row['Avg Price']:.2f}"
            ])
        
        print(tabulate(revenue_data, headers=[
            "Rank", "Product", "Category", "Units Sold", "Net Sales", "Avg Price"
        ], tablefmt="grid"))
        
        # Display bottom products by units sold
        print("\nBOTTOM 20 PRODUCTS BY UNITS SOLD (with sales > 0):")
        print("=" * 80)
        
        units_data = []
        for i, (_, row) in enumerate(bottom_by_units.iterrows(), 1):
            units_data.append([
                i, 
                row['Name'][:40], 
                row['Category Name'][:15], 
                row['Sold'], 
                f"${row['Net Sales']:.2f}", 
                f"${row['Avg Price']:.2f}"
            ])
        
        print(tabulate(units_data, headers=[
            "Rank", "Product", "Category", "Units Sold", "Net Sales", "Avg Price"
        ], tablefmt="grid"))
        
        # Count products with zero sales
        all_products = set()
        zero_sales_products = set()
        
        for report in parsed_reports:
            df = report['data']
            current_products = set(zip(df['Name'], df['Category Name']))
            
            for prod in current_products:
                product_name, category = prod
                all_products.add((product_name, category))
                
                # Check if this product has zero sales in this report
                product_data = df[(df['Name'] == product_name) & (df['Category Name'] == category)]
                if product_data['Sold'].sum() == 0:
                    zero_sales_products.add(prod)
        
        print(f"\nTotal Products with Zero Sales: {len(zero_sales_products)}")
        
        input("\nPress Enter to continue...")
    
    def sales_by_category(self):
        """Display sales broken down by category"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        # Offer option to choose a specific month or view combined data
        print("\nSALES BY CATEGORY:")
        print("1. View Combined Category Data")
        print("2. View Category Data for a Specific Month")
        
        choice = input("\nSelect an option: ")
        
        if choice == '1':
            # Parse all reports
            parsed_reports = []
            for report_file in reports:
                report = self.parse_report(report_file)
                if report:
                    parsed_reports.append(report)
            
            if not parsed_reports:
                print("Error parsing reports.")
                input("\nPress Enter to continue...")
                return
            
            # Combine all data and filter out the "TOTAL" category
            all_data = pd.concat([report['data'] for report in parsed_reports])
            all_data_filtered = all_data[all_data['Category Name'] != 'TOTAL']
            total_net_sales = all_data_filtered['Net Sales'].sum()
            
            # Group by category
            category_sales = all_data_filtered.groupby('Category Name').agg({
                'Sold': 'sum',
                'Net Sales': 'sum'
            }).sort_values('Net Sales', ascending=False).reset_index()
            
            # Calculate percentages
            category_sales['% of Total'] = (category_sales['Net Sales'] / total_net_sales) * 100
            
            # Display results
            print("\nCATEGORY SALES SUMMARY (ALL REPORTS COMBINED):")
            print("=" * 80)
            
            table_data = []
            for _, row in category_sales.iterrows():
                table_data.append([
                    row['Category Name'], 
                    row['Sold'], 
                    f"${row['Net Sales']:.2f}", 
                    f"{row['% of Total']:.2f}%"
                ])
            
            print(tabulate(table_data, headers=[
                "Category", "Units Sold", "Net Sales", "% of Total"
            ], tablefmt="grid"))
            
        elif choice == '2':
            # Choose a specific month
            print("\nAvailable Monthly Reports:")
            for i, report_file in enumerate(reports, 1):
                report = self.parse_report(report_file)
                if report:
                    print(f"{i}. {report['month']} {report['year']} - {report_file}")
            
            try:
                report_choice = int(input("\nSelect a report: "))
                if 1 <= report_choice <= len(reports):
                    report = self.parse_report(reports[report_choice-1])
                    if report:
                        df = report['data']
                        total_net_sales = df['Net Sales'].sum()
                        
                        # Group by category
                        category_sales = df.groupby('Category Name').agg({
                            'Sold': 'sum',
                            'Net Sales': 'sum'
                        }).sort_values('Net Sales', ascending=False).reset_index()
                        
                        # Calculate percentages
                        category_sales['% of Total'] = (category_sales['Net Sales'] / total_net_sales) * 100
                        
                        # Display results
                        print(f"\nCATEGORY SALES FOR {report['month']} {report['year']}:")
                        print("=" * 80)
                        
                        table_data = []
                        for _, row in category_sales.iterrows():
                            table_data.append([
                                row['Category Name'], 
                                row['Sold'], 
                                f"${row['Net Sales']:.2f}", 
                                f"{row['% of Total']:.2f}%"
                            ])
                        
                        print(tabulate(table_data, headers=[
                            "Category", "Units Sold", "Net Sales", "% of Total"
                        ], tablefmt="grid"))
                        
                        # Show top products in each category
                        for category in category_sales['Category Name']:
                            cat_products = df[df['Category Name'] == category].sort_values('Net Sales', ascending=False).head(5)
                            
                            if not cat_products.empty:
                                print(f"\nTop 5 Products in {category}:")
                                
                                cat_data = []
                                for i, (_, row) in enumerate(cat_products.iterrows(), 1):
                                    cat_data.append([
                                        i,
                                        row['Name'][:40],
                                        row['Sold'],
                                        f"${row['Net Sales']:.2f}"
                                    ])
                                
                                print(tabulate(cat_data, headers=[
                                    "Rank", "Product", "Units Sold", "Net Sales"
                                ], tablefmt="simple"))
                    else:
                        print("Error parsing report.")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        else:
            print("Invalid option.")
        
        export_choice = input("\nExport category summary to CSV? (y/n): ")
        if export_choice.lower() == 'y':
            self.export_category_summary()
        else:
            input("\nPress Enter to continue...")
            
    def top_products_by_category(self):
        """Display top products within each category"""
        reports = self.get_available_reports()
        
        if not reports:
            print("\nNo sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        # Offer option to choose a specific month or view combined data
        print("\nTOP PRODUCTS BY CATEGORY:")
        print("1. View Combined Data")
        print("2. View Data for a Specific Month")
        
        choice = input("\nSelect an option: ")
        
        if choice == '1':
            # Parse all reports
            parsed_reports = []
            for report_file in reports:
                report = self.parse_report(report_file)
                if report:
                    parsed_reports.append(report)
            
            if not parsed_reports:
                print("Error parsing reports.")
                input("\nPress Enter to continue...")
                return
            
            # Combine all data and filter out the "TOTAL" category
            all_data = pd.concat([report['data'] for report in parsed_reports])
            all_data_filtered = all_data[all_data['Category Name'] != 'TOTAL']
            
            # Get unique categories
            categories = all_data_filtered['Category Name'].unique()
            
            # Display results
            for category in sorted(categories):
                # Get data for this category
                category_data = all_data_filtered[all_data_filtered['Category Name'] == category]
                
                # Skip if empty
                if category_data.empty:
                    continue
                
                # Group by product and sum sales
                products = category_data.groupby('Name').agg({
                    'Sold': 'sum',
                    'Net Sales': 'sum'
                }).reset_index()
                
                # Sort by revenue and get top 10
                top_products = products.sort_values('Net Sales', ascending=False).head(10)
                
                # Display top products for this category
                print(f"\n\nTOP 10 PRODUCTS IN {category.upper()} CATEGORY:")
                print("=" * 80)
                
                if top_products.empty:
                    print("No products found in this category.")
                    continue
                
                table_data = []
                for i, (_, row) in enumerate(top_products.iterrows(), 1):
                    name_val = str(row['Name']) if not pd.isna(row['Name']) else "Unknown"
                    name = name_val[:40] if len(name_val) > 40 else name_val
                    
                    table_data.append([
                        i,
                        name,
                        row['Sold'],
                        f"${row['Net Sales']:.2f}",
                        f"${row['Net Sales']/row['Sold']:.2f}" if row['Sold'] > 0 else "$0.00"
                    ])
                
                print(tabulate(table_data, headers=[
                    "Rank", "Product Name", "Units Sold", "Net Sales", "Avg Price"
                ], tablefmt="grid"))
                
        elif choice == '2':
            # Choose a specific month
            print("\nAvailable Monthly Reports:")
            for i, report_file in enumerate(reports, 1):
                report = self.parse_report(report_file)
                if report:
                    print(f"{i}. {report['month']} {report['year']} - {report_file}")
            
            try:
                report_choice = int(input("\nSelect a report: "))
                if 1 <= report_choice <= len(reports):
                    report = self.parse_report(reports[report_choice-1])
                    
                    if report:
                        # Filter out the "TOTAL" category
                        df = report['data']
                        df_filtered = df[df['Category Name'] != 'TOTAL']
                        
                        # Get unique categories
                        categories = df_filtered['Category Name'].unique()
                        
                        # Display results for each category
                        for category in sorted(categories):
                            # Get data for this category
                            category_data = df_filtered[df_filtered['Category Name'] == category]
                            
                            # Skip if empty
                            if category_data.empty:
                                continue
                            
                            # Sort by revenue and get top 10
                            top_products = category_data.sort_values('Net Sales', ascending=False).head(10)
                            
                            # Display top products for this category
                            print(f"\n\nTOP 10 PRODUCTS IN {category.upper()} CATEGORY - {report['month']} {report['year']}:")
                            print("=" * 80)
                            
                            if top_products.empty:
                                print("No products found in this category.")
                                continue
                            
                            table_data = []
                            for i, (_, row) in enumerate(top_products.iterrows(), 1):
                                name_val = str(row['Name']) if not pd.isna(row['Name']) else "Unknown"
                                name = name_val[:40] if len(name_val) > 40 else name_val
                                
                                table_data.append([
                                    i,
                                    name,
                                    row['Sold'],
                                    f"${row['Net Sales']:.2f}",
                                    f"${row['Net Sales']/row['Sold']:.2f}" if row['Sold'] > 0 else "$0.00"
                                ])
                            
                            print(tabulate(table_data, headers=[
                                "Rank", "Product Name", "Units Sold", "Net Sales", "Avg Price"
                            ], tablefmt="grid"))
                    else:
                        print("Error parsing report.")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        else:
            print("Invalid option.")
        
        input("\nPress Enter to continue...")
    
    def compare_categories(self):
        """Compare categories across different months"""
        reports = self.get_available_reports()
        
        if len(reports) < 2:
            print("\nAt least 2 reports are needed for category comparison.")
            input("\nPress Enter to continue...")
            return
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if len(parsed_reports) < 2:
            print("Not enough valid reports for category comparison.")
            input("\nPress Enter to continue...")
            return
        
        # Sort reports by date
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        def get_sort_key(report):
            month = report['month']
            year = int(report['year'])
            month_num = month_order.get(month[:3], 0)
            return (year, month_num)
        
        sorted_reports = sorted(parsed_reports, key=get_sort_key)
        
        # Get all unique categories
        all_categories = set()
        for report in sorted_reports:
            df = report['data']
            df_filtered = df[df['Category Name'] != 'TOTAL']
            categories = df_filtered['Category Name'].unique()
            all_categories.update(categories)
        
        # Choose categories to compare
        print("\nAvailable Categories:")
        for i, category in enumerate(sorted(all_categories), 1):
            print(f"{i}. {category}")
        
        selections = input("\nSelect categories to compare (comma-separated numbers, or 'all' for all categories): ")
        
        if selections.lower() == 'all':
            selected_categories = sorted(all_categories)
        else:
            try:
                category_list = sorted(list(all_categories))
                indices = [int(idx.strip()) - 1 for idx in selections.split(',')]
                selected_categories = [category_list[i] for i in indices if 0 <= i < len(category_list)]
                
                if not selected_categories:
                    print("No valid categories selected.")
                    input("\nPress Enter to continue...")
                    return
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers or 'all'.")
                input("\nPress Enter to continue...")
                return
        
        # Create a dictionary to store category data for each month
        category_data = {}
        months = []
        
        for report in sorted_reports:
            month_label = f"{report['month']} {report['year']}"
            months.append(month_label)
            
            df = report['data']
            df_filtered = df[df['Category Name'] != 'TOTAL']
            
            for category in selected_categories:
                if category not in category_data:
                    category_data[category] = {
                        'units': [],
                        'sales': [],
                        'percent': []
                    }
                
                # Get data for this category in this month
                cat_df = df_filtered[df_filtered['Category Name'] == category]
                total_month_sales = df_filtered['Net Sales'].sum()
                
                if not cat_df.empty:
                    units = cat_df['Sold'].sum()
                    sales = cat_df['Net Sales'].sum()
                    percent = (sales / total_month_sales * 100) if total_month_sales > 0 else 0
                else:
                    units = 0
                    sales = 0
                    percent = 0
                
                category_data[category]['units'].append(units)
                category_data[category]['sales'].append(sales)
                category_data[category]['percent'].append(percent)
        
        # Display comparison by total revenue
        print("\nCATEGORY COMPARISON BY REVENUE:")
        print("=" * 80)
        
        # Prepare headers
        headers = ["Category"]
        for month in months:
            headers.append(month)
        
        # Prepare table data
        revenue_data = []
        for category in selected_categories:
            row = [category]
            for sales in category_data[category]['sales']:
                row.append(f"${sales:.2f}")
            revenue_data.append(row)
        
        # Sort by total revenue
        total_revenue = {cat: sum(category_data[cat]['sales']) for cat in selected_categories}
        revenue_data.sort(key=lambda x: total_revenue[x[0]], reverse=True)
        
        print(tabulate(revenue_data, headers=headers, tablefmt="grid"))
        
        # Display comparison by percentage of monthly sales
        print("\nCATEGORY COMPARISON BY PERCENTAGE OF MONTHLY SALES:")
        print("=" * 80)
        
        percent_data = []
        for category in selected_categories:
            row = [category]
            for percent in category_data[category]['percent']:
                row.append(f"{percent:.2f}%")
            percent_data.append(row)
        
        # Sort by average percentage
        avg_percent = {cat: sum(category_data[cat]['percent']) / len(months) for cat in selected_categories}
        percent_data.sort(key=lambda x: avg_percent[x[0]], reverse=True)
        
        print(tabulate(percent_data, headers=headers, tablefmt="grid"))
        
        # Display growth rates for top categories
        print("\nMONTH-OVER-MONTH GROWTH BY CATEGORY:")
        print("=" * 80)
        
        # Get top 5 categories by total revenue
        top_categories = sorted(selected_categories, key=lambda x: total_revenue[x], reverse=True)[:5]
        
        for category in top_categories:
            print(f"\n{category} Category Growth:")
            
            growth_data = []
            for i in range(1, len(months)):
                period = f"{months[i-1]} to {months[i]}"
                sales_prev = category_data[category]['sales'][i-1]
                sales_curr = category_data[category]['sales'][i]
                
                if sales_prev > 0:
                    growth_pct = ((sales_curr - sales_prev) / sales_prev) * 100
                    growth_data.append([period, f"{growth_pct:.2f}%"])
                else:
                    growth_data.append([period, "N/A"])
            
            print(tabulate(growth_data, headers=["Period", "Revenue Growth"], tablefmt="simple"))
        
        # Option to export
        export_choice = input("\nExport category comparison to CSV? (y/n): ")
        if export_choice.lower() == 'y':
            try:
                # Create output filename
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"category_comparison_{timestamp}.csv"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', newline='') as file:
                    writer = csv.writer(file)
                    
                    # Write header
                    writer.writerow(["Category Comparison", ", ".join(months)])
                    writer.writerow([])
                    
                    # Write revenue data
                    writer.writerow(["Category Revenue"] + months)
                    for category in selected_categories:
                        writer.writerow([category] + [f"${sales:.2f}" for sales in category_data[category]['sales']])
                    
                    writer.writerow([])
                    
                    # Write percentage data
                    writer.writerow(["Category Percentage"] + months)
                    for category in selected_categories:
                        writer.writerow([category] + [f"{percent:.2f}%" for percent in category_data[category]['percent']])
                    
                    writer.writerow([])
                    
                    # Write growth data
                    writer.writerow(["Category Growth"])
                    writer.writerow(["Category"] + [f"{months[i-1]} to {months[i]}" for i in range(1, len(months))])
                    
                    for category in selected_categories:
                        growth_values = []
                        for i in range(1, len(months)):
                            sales_prev = category_data[category]['sales'][i-1]
                            sales_curr = category_data[category]['sales'][i]
                            
                            if sales_prev > 0:
                                growth_pct = ((sales_curr - sales_prev) / sales_prev) * 100
                                growth_values.append(f"{growth_pct:.2f}%")
                            else:
                                growth_values.append("N/A")
                        
                        writer.writerow([category] + growth_values)
                
                print(f"\nCategory comparison exported to {filepath}")
            except Exception as e:
                print(f"\nError exporting comparison: {e}")
        
        input("\nPress Enter to continue...")
    
    def category_growth_analysis(self):
        """Analyze growth trends by category"""
        reports = self.get_available_reports()
        
        if len(reports) < 2:
            print("\nAt least 2 reports are needed for growth analysis.")
            input("\nPress Enter to continue...")
            return
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if len(parsed_reports) < 2:
            print("Not enough valid reports for growth analysis.")
            input("\nPress Enter to continue...")
            return
        
        # Sort reports by date
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        def get_sort_key(report):
            month = report['month']
            year = int(report['year'])
            month_num = month_order.get(month[:3], 0)
            return (year, month_num)
        
        sorted_reports = sorted(parsed_reports, key=get_sort_key)
        
        # Get all unique categories
        all_categories = set()
        for report in sorted_reports:
            df = report['data']
            df_filtered = df[df['Category Name'] != 'TOTAL']
            categories = df_filtered['Category Name'].unique()
            all_categories.update(categories)
        
        # Combine all data to find top categories overall
        all_data = pd.concat([report['data'] for report in sorted_reports])
        all_data_filtered = all_data[all_data['Category Name'] != 'TOTAL']
        
        # Calculate total revenue for each category across all periods
        category_totals = all_data_filtered.groupby('Category Name')['Net Sales'].sum().sort_values(ascending=False)
        top_categories = category_totals.head(5).index.tolist()
        
        # Calculate monthly data for each category
        months = [f"{report['month']} {report['year']}" for report in sorted_reports]
        category_data = {}
        
        for category in all_categories:
            category_data[category] = {
                'units': [],
                'sales': [],
                'percent': []
            }
            
            for report in sorted_reports:
                df = report['data']
                df_filtered = df[df['Category Name'] != 'TOTAL']
                monthly_total = df_filtered['Net Sales'].sum()
                
                # Get category data for this month
                cat_df = df_filtered[df_filtered['Category Name'] == category]
                
                if not cat_df.empty:
                    units = cat_df['Sold'].sum()
                    sales = cat_df['Net Sales'].sum()
                    percent = (sales / monthly_total * 100) if monthly_total > 0 else 0
                else:
                    units = 0
                    sales = 0
                    percent = 0
                
                category_data[category]['units'].append(units)
                category_data[category]['sales'].append(sales)
                category_data[category]['percent'].append(percent)
        
        # Calculate growth rates for each category
        print("\nCATEGORY GROWTH ANALYSIS")
        print("=" * 80)
        
        print("\nTop 5 Categories by Overall Revenue:")
        print(f"{'Category':<20}| {'Total Revenue':<15}| {'Avg Monthly':<15}| {'Trend':<15}")
        print("-" * 70)
        
        for category in top_categories:
            total_rev = category_totals[category]
            avg_monthly = total_rev / len(sorted_reports)
            
            # Determine trend based on first vs last month
            first_month = category_data[category]['sales'][0]
            last_month = category_data[category]['sales'][-1]
            
            if first_month > 0:
                change = ((last_month - first_month) / first_month) * 100
                if change > 5:
                    trend = "↑ Growing"
                elif change < -5:
                    trend = "↓ Declining"
                else:
                    trend = "→ Stable"
            else:
                trend = "N/A"
            
            print(f"{category:<20}| ${total_rev:<13.2f}| ${avg_monthly:<13.2f}| {trend:<15}")
        
        # Show growth details for each top category
        print("\nMonthly Revenue for Top Categories:")
        print("=" * 80)
        
        # Create data table
        headers = ["Category"] + months
        data = []
        
        for category in top_categories:
            row = [category]
            for sales in category_data[category]['sales']:
                row.append(f"${sales:.2f}")
            data.append(row)
        
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
        # Show growth rates
        print("\nMonth-over-Month Growth Rates:")
        print("=" * 80)
        
        for category in top_categories:
            print(f"\n{category} Category Growth:")
            
            growth_data = []
            for i in range(1, len(months)):
                period = f"{months[i-1]} to {months[i]}"
                sales_prev = category_data[category]['sales'][i-1]
                sales_curr = category_data[category]['sales'][i]
                
                if sales_prev > 0:
                    growth_pct = ((sales_curr - sales_prev) / sales_prev) * 100
                    status = "↑" if growth_pct > 0 else "↓" if growth_pct < 0 else "→"
                    growth_data.append([period, f"{growth_pct:.2f}%", status])
                else:
                    growth_data.append([period, "N/A", ""])
            
            print(tabulate(growth_data, headers=["Period", "Growth Rate", ""], tablefmt="simple"))
        
        # Show category share over time
        print("\nCategory Share of Monthly Revenue (%):")
        print("=" * 80)
        
        share_data = []
        for category in top_categories:
            row = [category]
            for percent in category_data[category]['percent']:
                row.append(f"{percent:.2f}%")
            share_data.append(row)
        
        print(tabulate(share_data, headers=headers, tablefmt="grid"))
        
        input("\nPress Enter to continue...")
    
    def monthly_growth_analysis(self):
        """Analyze month-over-month growth trends"""
        reports = self.get_available_reports()
        
        if len(reports) < 2:
            print("\nAt least 2 reports are needed for growth analysis.")
            input("\nPress Enter to continue...")
            return
        
        # Parse all reports
        parsed_reports = []
        for report_file in reports:
            report = self.parse_report(report_file)
            if report:
                parsed_reports.append(report)
        
        if len(parsed_reports) < 2:
            print("Not enough valid reports for growth analysis.")
            input("\nPress Enter to continue...")
            return
        
        # Sort reports by date
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        def get_sort_key(report):
            month = report['month']
            year = int(report['year'])
            month_num = month_order.get(month[:3], 0)
            return (year, month_num)
        
        sorted_reports = sorted(parsed_reports, key=get_sort_key)
        
        # Calculate monthly totals and growth
        months = []
        total_units = []
        total_sales = []
        
        for report in sorted_reports:
            df = report['data']
            # Filter out the "TOTAL" category to avoid double counting
            df_filtered = df[df['Category Name'] != 'TOTAL']
            
            month_label = f"{report['month']} {report['year']}"
            units = df_filtered['Sold'].sum()
            sales = df_filtered['Net Sales'].sum()
            
            months.append(month_label)
            total_units.append(units)
            total_sales.append(sales)
        
        # Calculate growth percentages
        units_growth = []
        sales_growth = []
        
        for i in range(1, len(sorted_reports)):
            if total_units[i-1] > 0:
                unit_growth = ((total_units[i] - total_units[i-1]) / total_units[i-1]) * 100
                units_growth.append(f"{unit_growth:.2f}%")
            else:
                units_growth.append("N/A")
            
            if total_sales[i-1] > 0:
                sale_growth = ((total_sales[i] - total_sales[i-1]) / total_sales[i-1]) * 100
                sales_growth.append(f"{sale_growth:.2f}%")
            else:
                sales_growth.append("N/A")
        
        # Display monthly totals
        print("\nMONTHLY SALES TOTALS:")
        print("=" * 80)
        
        data = []
        for i, month in enumerate(months):
            data.append([
                month,
                total_units[i],
                f"${total_sales[i]:.2f}"
            ])
        
        print(tabulate(data, headers=[
            "Month", "Units Sold", "Net Sales"
        ], tablefmt="grid"))
        
        # Display growth rates
        if len(sorted_reports) >= 2:
            print("\nMONTH-OVER-MONTH GROWTH RATES:")
            print("=" * 80)
            
            growth_data = []
            for i in range(len(units_growth)):
                period = f"{months[i]} to {months[i+1]}"
                growth_data.append([
                    period,
                    units_growth[i],
                    sales_growth[i]
                ])
            
            print(tabulate(growth_data, headers=[
                "Period", "Units Growth", "Sales Growth"
            ], tablefmt="grid"))
        
        # Category growth analysis
        print("\nCATEGORY GROWTH ANALYSIS:")
        print("=" * 80)
        
        # Get all unique categories
        all_categories = set()
        for report in sorted_reports:
            categories = report['data']['Category Name'].unique()
            all_categories.update(categories)
        
        # Select top categories by overall sales
        all_data = pd.concat([report['data'] for report in sorted_reports])
        category_totals = all_data.groupby('Category Name')['Net Sales'].sum().sort_values(ascending=False)
        top_categories = category_totals.head(5).index.tolist()
        
        # Display growth for top categories
        for category in top_categories:
            print(f"\nSales Trend for Category: {category}")
            
            cat_data = []
            cat_sales = []
            
            for i, report in enumerate(sorted_reports):
                df = report['data']
                cat_df = df[df['Category Name'] == category]
                cat_sale = cat_df['Net Sales'].sum()
                cat_units = cat_df['Sold'].sum()
                
                cat_data.append([
                    months[i],
                    cat_units,
                    f"${cat_sale:.2f}"
                ])
                
                cat_sales.append(cat_sale)
            
            print(tabulate(cat_data, headers=[
                "Month", "Units Sold", "Net Sales"
            ], tablefmt="simple"))
            
            # Calculate and display growth
            if len(sorted_reports) >= 2:
                print("\nGrowth Rates:")
                growth_data = []
                
                for i in range(1, len(sorted_reports)):
                    period = f"{months[i-1]} to {months[i]}"
                    if cat_sales[i-1] > 0:
                        growth_pct = ((cat_sales[i] - cat_sales[i-1]) / cat_sales[i-1]) * 100
                        growth_data.append([period, f"{growth_pct:.2f}%"])
                    else:
                        growth_data.append([period, "N/A"])
                
                print(tabulate(growth_data, headers=[
                    "Period", "Sales Growth"
                ], tablefmt="simple"))
        
        input("\nPress Enter to continue...")
