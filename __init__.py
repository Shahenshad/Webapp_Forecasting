from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.arima.model import ARIMAResults
import smtplib
from email.mime.text import MIMEText
app = Flask(__name__)

# Load your data
df = pd.read_csv("application/models/dataset.csv")
df['date'] = pd.to_datetime(df['date'])
df['product_identifier'] = df['product_identifier'].astype('category')
df['department_identifier'] = df['department_identifier'].astype('category')
df['category_of_product'] = df['category_of_product'].astype('category')
df['outlet'] = df['outlet'].astype('category')
df['state'] = df['state'].astype('category')
df['sales'] = df['sales'].astype('int64')
df['week_id'] = df['week_id'].astype('category')

#Stock notofication
def check_and_notify_low_stock(df):
    low_stock_products = df.groupby("product_identifier")["sales"].sum()
    low_stock = low_stock_products[low_stock_products < 5]

    if not low_stock.empty:
        product_list = "\n".join(f"Product ID: {idx} - Total Sales: {qty}" for idx, qty in low_stock.items())
        msg = MIMEText(f"The following products are low in stock (sales < 5):\n\n{product_list}")
        msg["Subject"] = "Low Stock Alert"
        msg["From"] = "shahenshad10@gmail.com"
        msg["To"] = "shahenshad8@gmail.com"

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login("shahenshad10@gmail.com", "Saju@2018" \
                "")
                server.send_message(msg)
                print("Alert email sent to supplier.")
        except Exception as e:
            print("Email sending failed:", e)

# Trigger the email check after data is loaded
check_and_notify_low_stock(df)

df = df.set_index('date')
df_exog = pd.get_dummies(df[['state', 'category_of_product', 'outlet']], drop_first=True)
print(df.info())

# df = df.sort_index()

# Ensure 'sales' column exists
sales_series = df['sales'].copy()

with open("application/models/arima_model.pkl", "rb") as f:
    arima_model = pickle.load(f)

with open("application/models/sarima_model.pkl", "rb") as f:
    sarima_model = pickle.load(f)

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_forecast(pred_series, title="Forecast"):
    fig, ax = plt.subplots(figsize=(10, 4))
    pred_series.plot(ax=ax, color="blue", label="Forecast")
    ax.set_title(title)
    ax.legend()
    return plot_to_base64(fig)

def resample_and_plot(df, rule, title):
    resampled = df['sales'].resample(rule).sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    resampled.plot(ax=ax)
    ax.set_title(title)
    return plot_to_base64(fig)

def plot_sales_by(df, group_by_col, title):
    grouped = df.groupby([pd.Grouper(freq='M'), group_by_col])['sales'].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend(loc="best", fontsize='small')
    return plot_to_base64(fig)


@app.route('/', methods=['GET', 'POST'])
def index():
    plot, table = None, None
    daily_plot, weekly_plot, monthly_plot, yearly_plot = None, None, None, None

    if request.method == 'POST':
        n_days = int(request.form['n_days'])
        last_date = pd.to_datetime(request.form['last_date'])
        model_choice = request.form['model']

        # Generate future date range
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')

        if model_choice == "arima":
            forecast = arima_model.forecast(steps=n_days)
        else:
            forecast = sarima_model.forecast(steps=n_days, exog=df_exog.iloc[-n_days:])

        forecast.index = forecast_dates
        forecast_df = forecast.to_frame(name="Forecasted Sales")

        # Plot forecast
        plot = plot_forecast(forecast_df["Forecasted Sales"], title=f"{model_choice.upper()} Forecast")

        # Forecast table
        table = forecast_df.to_html(classes="table table-bordered", border=0)

    # Daily/Weekly/Monthly/Yearly plots
    df.index = pd.to_datetime(df.index)
    daily_plot = resample_and_plot(df, 'D', "Daily Sales")
    weekly_plot = resample_and_plot(df, 'W', "Weekly Sales")
    monthly_plot = resample_and_plot(df, 'M', "Monthly Sales")
    yearly_plot = resample_and_plot(df, 'Y', "Yearly Sales")

    product_plot = plot_sales_by(df, 'product_identifier', "Monthly Sales by Product ID")
    category_plot = plot_sales_by(df, 'category_of_product', "Monthly Sales by Category")
    department_plot = plot_sales_by(df, 'department_identifier', "Monthly Sales by Department")

    return render_template("index.html",
                           plot=plot,
                           table=table,
                           daily_plot=daily_plot,
                           weekly_plot=weekly_plot,
                           monthly_plot=monthly_plot,
                           yearly_plot=yearly_plot,
                           product_plot=product_plot,
                           category_plot=category_plot,
                           department_plot=department_plot)


