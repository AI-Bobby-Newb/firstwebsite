# Vape Store Business Analytics Dashboard

A comprehensive web-based analytics dashboard for vape store sales data visualization and business insights.

## Features

- **Dashboard Overview**: Get a quick overview of key metrics
- **Monthly Analysis**: Track sales performance over time
- **Product Analysis**: Analyze individual product performance
- **Category Analysis**: See which product categories drive your business
- **Top Products**: Identify your best-selling items
- **Business Insights**: Get actionable recommendations based on your data

## Deployment to Vercel

This application is configured to be easily deployed to Vercel for free hosting. Follow these steps to deploy:

### Prerequisites

1. [GitHub](https://github.com/) account
2. [Vercel](https://vercel.com/) account (you can sign up with your GitHub account)

### Step 1: Push to GitHub

1. Initialize a Git repository in the project folder (if not already done):
   ```bash
   git init
   ```

2. Add all files to Git:
   ```bash
   git add .
   ```

3. Commit the changes:
   ```bash
   git commit -m "Initial commit"
   ```

4. Create a new repository on GitHub.

5. Connect your local repository to GitHub and push:
   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   git push -u origin main
   ```

### Step 2: Deploy to Vercel

1. Go to [Vercel](https://vercel.com/) and sign in with your GitHub account.

2. Click "New Project" and select the GitHub repository you just created.

3. Vercel will automatically detect that it's a Flask application.

4. Configure the project:
   - Framework Preset: Other
   - Build Command: Leave empty (uses defaults)
   - Output Directory: Leave empty (uses defaults)
   - Install Command: `pip install -r requirements.txt`

5. Click "Deploy".

6. Vercel will build and deploy your application. Once complete, you'll get a URL where your application is hosted.

## Local Development

To run the application locally:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Data Structure

The application expects monthly sales data in CSV format stored in the `monthly_sales` directory. Each file should follow the format:

```
Category Name,Name,Sold,Net Sales
Category A,Product 1,10,100.00
Category B,Product 2,5,75.00
...
```

## Features

### Dashboard Overview
- Total sales, units sold, average price, and product count
- Monthly sales trends
- Top product categories
- Top products by revenue and units

### Monthly Analysis
- Month-over-month sales and unit growth
- Trend visualization
- Seasonal patterns identification

### Product Analysis
- Individual product performance metrics
- Category-specific filtering
- Unit sales vs. revenue analysis

### Category Analysis
- Performance breakdown by product category
- Market share visualization
- Category trend analysis

### Top Products
- Highest revenue generators
- Best-selling units
- Price point analysis

### Business Insights
- Data-driven business recommendations
- Revenue distribution visualization
- Product pricing tier analysis
- Performance matrix for strategic decision-making

## Technology Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Chart.js, Plotly
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Vercel

## License

This project is licensed under the MIT License - see the LICENSE file for details.
