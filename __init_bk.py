from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import pickle

app = Flask(__name__)

# Load your trained models
with open("application/models/arima_model.pkl", "rb") as f:
    arima_model = pickle.load(f)

with open("application/models/sarima_model.pkl", "rb") as f:
    sarima_model = pickle.load(f)

# Forecast function with optional model selection
def forecast_sales(model, last_date_str, n_days):
    last_date = pd.to_datetime(last_date_str)
    forecast = model.forecast(steps=n_days)
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
    forecast.index = forecast_index
    return forecast

# Function to create a plot and return as base64
def create_plot(series, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax, marker='o', linestyle='-', color='blue')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecasted Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_plot = None
    forecast_table = None
    if request.method == 'POST':
        n_days = int(request.form['n_days'])
        model_type = request.form['model']
        last_date = request.form['last_date']

        if model_type == 'arima':
            forecast = forecast_sales(arima_model, last_date, n_days)
        else:
            forecast = forecast_sales(sarima_model, last_date, n_days)

        forecast_plot = create_plot(forecast, f"{model_type.upper()} Forecast for {n_days} Days")
        forecast_table = forecast.to_frame(name='Forecasted Sales').to_html(classes='table table-striped')

    return render_template('index.html', plot=forecast_plot, table=forecast_table)


