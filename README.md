🚖 Uber Data Analytics — SQL + Python Project
A complete analytics project on 150,000+ Uber ride bookings using MySQL, Python, Pandas, Seaborn, Matplotlib, and Cohort Analysis.

📌 Project Overview
This project analyzes Uber ride data from NCR to understand:
Ride volume trends
Revenue performance
Payment behavior
Vehicle-type insights
Cancellation patterns
Customer ratings
Cohort-based retention

📂 Project Structure : Uber-Analytics/
│
├── analysis.ipynb                 # Main Python + SQL analysis notebook
├── analysis.html / analysis.pdf   # Exported reports with charts
│
├── cohortanalysis.py              # Full cohort retention analysis script
├── cohort_counts.csv              # Users per cohort month
├── cohort_retention_fraction.csv  # Retention % per cohort
│
├── uber analysis.html             # Extra analysis export
├── uber data analytics py.pdf     # PDF report
├── uber data analytics.docx       # Summary / written analysis
│
└── DATA/
    └── ncr_ride_bookings.csv      # Raw dataset (150k+ rows)

🛠 Tech Stack:
Python (Pandas, Matplotlib, Seaborn)
MySQL + SQLAlchemy
Jupyter Notebook
Cohort Analysis Methods

📊 Key Insights

Total rides: 150,000+
Go Sedan has the highest average rating
UPI generates the highest revenue share
Cancellations exist on both driver & customer side
Retention drops after month 1 (seen in cohort heatmap)
Pickup/Drop hotspots show clear user patterns

🧠 SQL Queries Used (Examples): 
SELECT COUNT(*) FROM ncr_ride_bookings;

SELECT `Vehicle Type`, SUM(`Booking Value`) 
FROM ncr_ride_bookings GROUP BY `Vehicle Type`;

SELECT `Payment Method`, SUM(`Booking Value`) 
FROM ncr_ride_bookings GROUP BY `Payment Method`;

SELECT `Vehicle Type`, AVG(`Customer Rating`)
FROM ncr_ride_bookings GROUP BY `Vehicle Type`;

📈 Visuals in Notebook
Revenue by Vehicle Type
Revenue by Payment Method
Completed vs Cancelled Rides
Average Rating Comparison
Payment Method Pie Chart
Cohort Retention Heatmap

