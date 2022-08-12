#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
# from matplotlib import pyplot as plt
# from fbprophet import Prophet
from fbprophet.plot import plot_plotly

import streamlit as st  # pylint: disable=import-error

#FILEPATH = os.path.join(os.getcwd(), "app", "data.json")

ALL = "All Products - No Forecast"
PRODUCT_A = "Product_A"
PRODUCT_B = "Product_B"
PRODUCT_C = "Product_C"


@st.cache
def load_data(path):
    """Loads the dataset from a filepath."""
    return (
        pd.read_json(path)
        .rename(
            columns={
                "Total Results as of Date": "date",
                "Cases": "cumulative_cases",
                "Deaths": "cumulative_deaths",
                "Recovered": "cumulative_recoveries",
            }
        )
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index("date")
    )


@st.cache(allow_output_mutation=True)
def make_forecast(selection):
    """Takes a name from the selection and makes a forecast plot."""

    if selection == PRODUCT_A:

        cumulative_series_name = "cumulative_cases"
        title = "Forecast Product A"
        x_label = "Prices"

    if selection == PRODUCT_B:

        cumulative_series_name = "cumulative_deaths"
        title = "Forecast Product B"
        x_label = "Prices"

    if selection == PRODUCT_C:

        cumulative_series_name = "cumulative_recoveries"
        title = "Forecast Product C"
        x_label = "Prices"

    prophet_df = (
        df[cumulative_series_name]
        .diff()
        .dropna()
        .to_frame()
        .reset_index()
        .rename(columns={"date": "ds", cumulative_series_name: "y"})
    )

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title=title, yaxis_title=x_label, xaxis_title="Date",
    )

    return fig


df = load_data('data.json')
st.write("# Forecast Prices")

selected_series = st.selectbox("Select a data set:", (ALL, PRODUCT_A, PRODUCT_B, PRODUCT_C))

if selected_series == ALL:
    cases_series = df["cumulative_cases"]
    deaths_series = df["cumulative_deaths"]
    recoveries_series = df["cumulative_recoveries"]

    plt.title("Global Products")
    plt.xlabel("Date")
    plt.ylabel("Prices")
    plt.plot(cases_series.index, cases_series.values, label=PRODUCT_A)
    plt.plot(deaths_series.index, deaths_series.values, label=PRODUCT_B)
    plt.plot(recoveries_series.index, recoveries_series.values, label=PRODUCT_C)
    plt.legend()

    st.pyplot()

else:
    plotly_fig = make_forecast(selected_series)
    st.plotly_chart(plotly_fig)

