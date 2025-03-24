from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('business_insights.html', 
                           total_revenue=0,
                           total_units=0,
                           avg_price=0,
                           total_products=0,
                           product_metrics=[],
                           months=[],
                           years_dict={},
                           quality_score=100,
                           quality_warnings=[])

@app.route('/business_insights')
def business_insights():
    return render_template('business_insights.html',
                           total_revenue=0,
                           total_units=0, 
                           avg_price=0,
                           total_products=0,
                           product_metrics=[],
                           months=[],
                           years_dict={},
                           quality_score=100,
                           quality_warnings=[])

if __name__ == '__main__':
    app.run(debug=True)