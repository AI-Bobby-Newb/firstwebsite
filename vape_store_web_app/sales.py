import os
import csv
from datetime import datetime
import pandas as pd

class SalesManager:
    """Manages sales operations for the vape store system"""
    
    def __init__(self, db):
        """Initialize with database connection"""
        self.db = db
        self.monthly_sales_dir = "monthly_sales"
    
    def get_available_reports(self):
        """Get a list of available monthly sales reports"""
        reports = []
        
        if os.path.exists(self.monthly_sales_dir):
            for filename in os.listdir(self.monthly_sales_dir):
                if filename.endswith('.csv'):
                    reports.append(filename)
        
        return sorted(reports)
    
    def parse_sales_report(self, report_file):
        """Parse a monthly sales report CSV file and return the data"""
        filepath = os.path.join(self.monthly_sales_dir, report_file)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            # Read the CSV file
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Extract report header info
            report_title = lines[0].strip().split(',')[0].replace('"', '')
            report_period = lines[1].strip().split(',')[0].replace('"', '')
            
            # Parse the data using pandas
            df = pd.read_csv(filepath, skiprows=2)
            
            # Return a dictionary with report info and data
            return {
                'title': report_title,
                'period': report_period,
                'filename': report_file,
                'data': df
            }
        except Exception as e:
            print(f"Error parsing sales report: {e}")
            return None
    
    def view_recent_sales(self):
        """Display available monthly sales reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("No sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAvailable Monthly Sales Reports:")
        print("-" * 60)
        
        for i, report in enumerate(reports, 1):
            print(f"{i}. {report}")
        
        print("\n1. View Report Details")
        print("0. Back")
        
        choice = input("\nSelect an option: ")
        
        if choice == '1':
            report_num = input("Enter report number to view: ")
            
            try:
                report_idx = int(report_num) - 1
                if 0 <= report_idx < len(reports):
                    self.view_sales_report(reports[report_idx])
                else:
                    print("Invalid report number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice != '0':
            print("Invalid option.")
    
    def view_sales_report(self, report_file):
        """Display details of a specific sales report"""
        report = self.parse_sales_report(report_file)
        
        if not report:
            print(f"Error: Could not load report {report_file}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{report['title']}")
        print(f"Period: {report['period']}")
        print("-" * 60)
        
        # Get data from the report
        df = report['data']
        
        # Calculate summary statistics
        total_items_sold = df['Sold'].astype(int).sum()
        total_gross_sales = df['Gross Sales'].str.replace('$', '').str.replace(',', '').astype(float).sum()
        total_net_sales = df['Net Sales'].str.replace('$', '').str.replace(',', '').astype(float).sum()
        
        # Get top 10 selling items by units sold
        top_items_by_units = df.sort_values('Sold', ascending=False).head(10)
        
        # Get top 10 selling items by revenue
        df['Net Sales Numeric'] = df['Net Sales'].str.replace('$', '').str.replace(',', '').astype(float)
        top_items_by_revenue = df.sort_values('Net Sales Numeric', ascending=False).head(10)
        
        # Get sales by category
        category_sales = df.groupby('Category Name').agg({
            'Sold': 'sum',
            'Net Sales Numeric': 'sum'
        }).sort_values('Net Sales Numeric', ascending=False)
        
        # Display summary
        print(f"Total Items Sold: {total_items_sold}")
        print(f"Total Gross Sales: ${total_gross_sales:.2f}")
        print(f"Total Net Sales: ${total_net_sales:.2f}")
        
        # Display top items by units sold
        print("\nTop 10 Items by Units Sold:")
        print(f"{'#':<3}| {'Name':<40}| {'Category':<15}| {'Units':<6}| {'Sales':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(top_items_by_units.iterrows(), 1):
            name = row['Name'][:38] if len(row['Name']) > 38 else row['Name']
            category = row['Category Name'][:13] if len(row['Category Name']) > 13 else row['Category Name']
            print(f"{i:<3}| {name:<40}| {category:<15}| {row['Sold']:<6}| {row['Net Sales']:<10}")
        
        # Display sales by category
        print("\nSales by Category:")
        print(f"{'Category':<20}| {'Units Sold':<10}| {'Net Sales':<12}| {'% of Total':<10}")
        print("-" * 60)
        
        for category, row in category_sales.iterrows():
            category_name = category[:18] if len(category) > 18 else category
            percent = (row['Net Sales Numeric'] / total_net_sales) * 100
            print(f"{category_name:<20}| {row['Sold']:<10}| ${row['Net Sales Numeric']:<10.2f}| {percent:<9.2f}%")
        
        # Options
        print("\n1. Export to CSV")
        print("2. View All Items")
        print("0. Back")
        
        choice = input("\nSelect an option: ")
        
        if choice == '1':
            self.export_report_summary(report)
        elif choice == '2':
            self.view_all_report_items(report)
        elif choice != '0':
            print("Invalid option.")
        
        input("\nPress Enter to continue...")
    
    def view_all_report_items(self, report):
        """Display all items in a sales report with pagination"""
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
                name = row['Name'][:38] if len(row['Name']) > 38 else row['Name']
                category = row['Category Name'][:13] if len(row['Category Name']) > 13 else row['Category Name']
                print(f"{i:<4}| {name:<40}| {category:<15}| {row['Sold']:<6}| {row['Net Sales']:<10}")
            
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
    
    def export_report_summary(self, report):
        """Export a summary of the sales report to a CSV file"""
        try:
            # Create reports directory if it doesn't exist
            if not os.path.exists('reports'):
                os.makedirs('reports')
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"sales_summary_{timestamp}.csv"
            filepath = os.path.join('reports', filename)
            
            # Get data
            df = report['data']
            total_items_sold = df['Sold'].astype(int).sum()
            total_gross_sales = df['Gross Sales'].str.replace('$', '').str.replace(',', '').astype(float).sum()
            total_net_sales = df['Net Sales'].str.replace('$', '').str.replace(',', '').astype(float).sum()
            
            # Calculate category sales
            df['Net Sales Numeric'] = df['Net Sales'].str.replace('$', '').str.replace(',', '').astype(float)
            category_sales = df.groupby('Category Name').agg({
                'Sold': 'sum',
                'Net Sales Numeric': 'sum'
            }).sort_values('Net Sales Numeric', ascending=False)
            
            # Get top items
            top_items = df.sort_values('Net Sales Numeric', ascending=False).head(20)
            
            # Write CSV file
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
                    percent = (row['Net Sales Numeric'] / total_net_sales) * 100
                    writer.writerow([
                        category, 
                        row['Sold'],
                        f"${row['Net Sales Numeric']:.2f}",
                        f"{percent:.2f}%"
                    ])
                
                writer.writerow([])
                
                # Write top items
                writer.writerow(["Top 20 Items by Revenue"])
                writer.writerow(["Rank", "Name", "Category", "Units Sold", "Net Sales"])
                
                for i, (_, row) in enumerate(top_items.iterrows(), 1):
                    writer.writerow([
                        i,
                        row['Name'],
                        row['Category Name'],
                        row['Sold'],
                        row['Net Sales']
                    ])
            
            print(f"Report summary exported to {filepath}")
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
    
    def search_sales(self):
        """Search for specific products or categories across sales reports"""
        reports = self.get_available_reports()
        
        if not reports:
            print("No sales reports found.")
            input("\nPress Enter to continue...")
            return
        
        search_term = input("\nEnter search term (product name or category): ").lower()
        
        if not search_term:
            print("Search term cannot be empty.")
            input("\nPress Enter to continue...")
            return
        
        print(f"\nSearching for '{search_term}' across {len(reports)} reports...")
        
        results = []
        
        for report_file in reports:
            report = self.parse_sales_report(report_file)
            
            if report:
                df = report['data']
                
                # Search in product names and categories
                matches = df[
                    df['Name'].str.lower().str.contains(search_term) | 
                    df['Category Name'].str.lower().str.contains(search_term)
                ]
                
                if not matches.empty:
                    # Add report info to each match
                    matches['Report'] = report['period']
                    results.append(matches)
        
        if not results:
            print(f"No matches found for '{search_term}'.")
            input("\nPress Enter to continue...")
            return
        
        # Combine all results
        all_results = pd.concat(results)
        
        # Display results
        print(f"\nFound {len(all_results)} matches for '{search_term}':")
        print(f"{'Report':<25}| {'Name':<40}| {'Category':<15}| {'Units':<6}| {'Sales':<10}")
        print("-" * 100)
        
        for _, row in all_results.iterrows():
            name = row['Name'][:38] if len(row['Name']) > 38 else row['Name']
            category = row['Category Name'][:13] if len(row['Category Name']) > 13 else row['Category Name']
            print(f"{row['Report']:<25}| {name:<40}| {category:<15}| {row['Sold']:<6}| {row['Net Sales']:<10}")
        
        input("\nPress Enter to continue...")
    
    def sales_menu(self):
        """Display sales management menu"""
        while True:
            print("\nSALES MANAGEMENT")
            print("1. View Monthly Sales Reports")
            print("2. Search Sales")
            print("3. Sales Trends Analysis")
            print("0. Back to Main Menu")
            
            choice = input("\nSelect an option: ")
            
            if choice == '1':
                self.view_recent_sales()
            elif choice == '2':
                self.search_sales()
            elif choice == '3':
                self.sales_trends_analysis()
            elif choice == '0':
                break
            else:
                print("Invalid option. Please try again.")
    
    def sales_trends_analysis(self):
        """Analyze sales trends across multiple reports"""
        reports = self.get_available_reports()
        
        if len(reports) < 2:
            print("Need at least 2 sales reports for trend analysis.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAnalyzing sales trends across available reports...")
        
        all_data = []
        for report_file in reports:
            report = self.parse_sales_report(report_file)
            if report:
                # Extract month/year from period
                period = report['period'].split(' - ')[0]
                month_year = ' '.join(period.split()[:2])
                
                # Add report info to data
                df = report['data'].copy()
                df['Period'] = month_year
                df['Net Sales Numeric'] = df['Net Sales'].str.replace('$', '').str.replace(',', '').astype(float)
                all_data.append(df)
        
        if not all_data:
            print("Failed to load sales data.")
            input("\nPress Enter to continue...")
            return
        
        # Combine all data
        combined_data = pd.concat(all_data)
        
        # Calculate monthly totals
        monthly_totals = combined_data.groupby('Period').agg({
            'Sold': 'sum',
            'Net Sales Numeric': 'sum'
        }).reset_index()
        
        # Sort by period (assuming consistent naming like "Jan 2025")
        try:
            monthly_totals['Sort Date'] = pd.to_datetime(monthly_totals['Period'], format='%b %Y')
            monthly_totals = monthly_totals.sort_values('Sort Date')
        except:
            # If date parsing fails, just use the data as is
            pass
        
        # Category trends
        category_trends = combined_data.groupby(['Period', 'Category Name']).agg({
            'Net Sales Numeric': 'sum'
        }).reset_index()
        
        # Display monthly totals
        print("\nMonthly Sales Totals:")
        print(f"{'Month':<15}| {'Items Sold':<15}| {'Net Sales':<15}")
        print("-" * 50)
        
        for _, row in monthly_totals.iterrows():
            print(f"{row['Period']:<15}| {row['Sold']:<15}| ${row['Net Sales Numeric']:<13.2f}")
        
        # Calculate month-over-month growth if possible
        if len(monthly_totals) >= 2:
            monthly_totals['Sales Growth'] = monthly_totals['Net Sales Numeric'].pct_change() * 100
            print("\nMonth-over-Month Growth:")
            print(f"{'Month':<15}| {'Growth %':<10}")
            print("-" * 30)
            
            for i, row in monthly_totals.iterrows():
                if i > 0:  # Skip first row (no previous month)
                    growth = row['Sales Growth']
                    growth_str = f"{growth:.2f}%" if not pd.isna(growth) else "N/A"
                    print(f"{row['Period']:<15}| {growth_str:<10}")
        
        # Get top categories across all periods
        top_categories = combined_data.groupby('Category Name')['Net Sales Numeric'].sum().nlargest(5).index
        
        # Display category trends for top categories
        print("\nTop Category Trends:")
        for category in top_categories:
            print(f"\n{category}:")
            cat_data = category_trends[category_trends['Category Name'] == category]
            
            # Sort by period
            try:
                cat_data['Sort Date'] = pd.to_datetime(cat_data['Period'], format='%b %Y')
                cat_data = cat_data.sort_values('Sort Date')
            except:
                pass
            
            for _, row in cat_data.iterrows():
                print(f"{row['Period']}: ${row['Net Sales Numeric']:.2f}")
        
        input("\nPress Enter to continue...")
    
    def new_sale(self):
        """Process a new sale (placeholder for POS functionality)"""
        print("\nThis is a placeholder for the POS system.")
        print("In a complete implementation, this would include:")
        print("- Item scanning or selection")
        print("- Quantity adjustment")
        print("- Discounts and promotions")
        print("- Payment processing")
        print("- Receipt generation")
        print("\nCurrently, sales data is imported from monthly reports.")
        
        input("\nPress Enter to continue...")
    
    def process_return(self):
        """Process a return or exchange (placeholder)"""
        print("\nThis is a placeholder for returns processing.")
        print("In a complete implementation, this would include:")
        print("- Looking up the original sale")
        print("- Processing a refund or exchange")
        print("- Adjusting inventory")
        print("- Generating return receipt")
        
        input("\nPress Enter to continue...")
    
    def manage_promotions(self):
        """Manage promotions and discounts (placeholder)"""
        print("\nThis is a placeholder for promotions management.")
        print("In a complete implementation, this would include:")
        print("- Creating new promotions")
        print("- Setting discount amounts")
        print("- Defining promotion validity periods")
        print("- Viewing active and upcoming promotions")
        
        input("\nPress Enter to continue...")
