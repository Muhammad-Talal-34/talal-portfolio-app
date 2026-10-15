# DemandCast (Bixi Soft)

A retail demand forecasting and inventory-planning platform I worked on at Bixi Soft.

## Tech stack
Python, FastAPI, MySQL, Pandas, and Scikit-learn. The platform's dashboard is in React.

## What it does
It uses past sales data to predict 7-day and 30-day product demand, compares the forecast with current stock, and flags stock-out risk. It is a decision-support tool: it gives advice and the business owner reviews it.

## What I worked on
- Cleaning and validating sales data
- Building forecasting features such as lag sales and rolling averages
- Developing REST APIs with FastAPI and Django, with MySQL storage for sales, inventory, and forecast data
- Data-processing pipelines for validation, transformation, and storage

## Accuracy notes
Don't claim I built every module of the platform; the wider plan also covers dashboards, inventory recommendations, reports, and deployment tooling. Don't state forecast accuracy numbers, clients, or business results — none are confirmed.
