import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
import os
import sys
import warnings 

from statsmodels.graphics.tsaplots import plot_acf

from utils import adfuller_test, kpss_test

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# sns.set_style({'axes.grid':False})


def plot_interactive_line(data, col, title=None):

	"""
	Function to plot an interactive plot using Plotly

	Parameters:
	-----------
	data : pandas.DataFrame
		Represents the dataset.

	col : str
		Represents the current or the target column

	title : str, optional; default:None
		Represents the title for the plot
	
	Returns:
	-------
	fig : plotly.Figure
		Represents the interactive figure
	"""

	orgn = go.Scatter(name='Original Data',
					x = data.index,
					y = data[col],
					mode='lines',
					line=dict(width=3, color='royalblue'))

	layout = dict(autosize=False, width=1000, title=title, 
		xaxis = dict(
			showgrid=False,
			rangeslider = dict(visible=True),
			rangeselector = dict(
				buttons = list([
					dict(count=1, label='1m', step='month', stepmode='backward'),
					dict(count=3, label='3m', step='month', stepmode='backward'),
					dict(count=6, label='6m', step='month', stepmode='backward'),
					dict(step='all')
					])
				)
			),

		yaxis = dict(automargin=True, showgrid=False))

	fig = go.Figure(orgn, layout=layout)

	return fig


def plot_lag_plots(data, col, lags=None):
	"""
	Function to plot lag plots for a specific value

	Parameters:
	-----------
	data : pandas.DataFrame
		Represents the dataframe with which we are working

	col : str
		Represents the target column for the data

	lags : array-like
		Represents the lags for the plot.

	Returns:
	--------
	fig : matplotlib.Figure
		Represents the lagplot

	"""

	if lags is None:
		return

	if len(lags) % 2 != 0:
		raise ValueError(f'Expected no of elements in lags to be even. Found {len(lags)}')


	fig, axes = plt.subplots(len(lags)//2, 2, figsize=(10, 4))
	plt.subplots_adjust(hspace=0.4, wspace=0.4, top=2.5)
	for i in range(len(lags)):
		
		pd.plotting.lag_plot(data[col], lag=lags[i], ax=axes[i//2, i%2], alpha=0.8)
		axes[i//2, i%2].set_title(f'Number of lags {lags[i]}')

	return fig


if __name__ == "__main__":

	df = pd.read_csv("./DATA/bank_nifty.csv", index_col=0)
	df = df.dropna(axis=0)
	
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
	df['Date'] = df['date'].dt.strftime('%Y-%m-%d').astype('str') + " " + df['time'].astype('str')
	df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')
	df.drop(['date', 'time'], inplace=True, axis=1)
	df = df.set_index('Date')
	df = df.sort_index()
	df = df.loc['2020-01-01 00:00:00':'2021-12-01 00:00:00']

	st.set_page_config(page_title='NIFTY', layout='wide')

	st.markdown(""" 
	<head>
	</head>
	<body>
	<div style='background-color:#272928; border-radius:25px; border:10px'>
	<h1 style='text-align:center; font-size:48px'> 💸 </h1>
	<h2 style='text-align:center; font-size:48px; font-weight:bold; color:#62b58b'> Analysis of NIFTY Data </h2>
	<p style='text-align:center; font-size:14px; color:#a8adaa; padding-bottom:25px'> An in-depth data dashboard of the Bank NIFTY data for the year 2020-2021.</p>
	</div>
	</body>

	""", unsafe_allow_html=True)
	
	rad = st.sidebar.radio("Navigation", ["Home", "Data Insights", "Statistical Tests"])

	if rad == "Home":
		st.header("Top 100 rows of the data")
		cols = ["open", "close"]
		st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)
		st.dataframe(df[st_ms].head(100))
		st.markdown(f"""The above DataFrame represents the Bank NIFTY data from the year 2020-2021. 
			Here each of the data is recorded at a 1 min interval, therefore producing a total records of {df.shape[0]}. The 
			data was collected from Kaggle.
			""")

		

		st.markdown(""" Our primary focus in this project is to understand the 
			behavior of the Closing price of the NIFTY Data for the year 2020-2021. """)

		st.markdown("")
		st.header("Percentage of Data Missing in each column")
		miss_df = pd.DataFrame((df.isna().sum() / df.isna().count())*100, columns=['percentage_missing'])
		st.dataframe(miss_df)
		st.markdown("Seems like there are no missing values in the dataframe")


		st.markdown("")
		st.header('Close Time Series')
		fig_close = plot_interactive_line(df, 'close')
		st.write(fig_close)
		
	if rad == "Data Insights":	
		st.markdown("")
		st.header("Distribution Plot of the Close Prices of NIFTY")
		fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8))
		sns.boxplot(x=df['close'], ax=ax1[0])
		sns.distplot(df['close'], ax=ax1[1])
		st.pyplot(fig1)
		st.markdown(""" The data represents a **multi-modal distribution** as per the above distribution plot. 
			Meaning that there more that one mode in the data.""")

		st.markdown("")
		st.header('Auto-Correlation Plot for the Close Price of NIFTY')
		fig2, ax2 = plt.subplots(figsize=(10, 4))
		plot_acf(df['close'], ax=ax2, lags=43200*2, use_vlines=False, title='ACF Plot for the close prices')
		st.pyplot(fig2)
		st.markdown(""" The Auto-Correlation suggests that **seasonality** is **absent** in the data. By seasonality we mean 
			a set of periodic behavior found in the data, where some behavior tends to repeat itself at a certain period.""")

		st.markdown("")
		st.header('Lag Plots for the Close Price of NIFTY')
		fig3 = plot_lag_plots(df, 'close', lags=[1, 5, 10, 30, 60, 720, 1440, 2880])
		st.pyplot(fig3)
		st.markdown(""" The lag plots suggest that there is a presence of strong positive 
			auto-correlation for a maximum of 60 mins lags, after which the auto-correlation starts to decrease.""")

	if rad == "Statistical Tests":
		st.markdown("")
		st.header('Results of the ADF Test')
		st.markdown(""" 
			One of the famous stationarity tests for time series is the Augmented Dickey-Fuller Test. The purpose of this test 
			is to measure whether a given time series is stationary or not. A Time Series is said to be stationary
			if and only if the trend component and seasonal component in the data is absent.
			""")

		st.markdown("*Note: The results may take time depending on your data length*")
		gif_runner = st.image('./Lazy-Loader/96x96.gif')
		adf_res, adf_is_stn = adfuller_test(df, 'close')
		gif_runner.empty()
		st.dataframe(adf_res)
		st.markdown(f"The result of the Augmented Dickey-Fuller test conveys that the time series being **stationary** is **{adf_is_stn}**")


		kpss_res, kpss_is_stn = kpss_test(df, 'close')
		
