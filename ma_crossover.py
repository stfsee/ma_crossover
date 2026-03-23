import pandas as pd
import yfinance as yf
import numpy as np
from prophet import Prophet
from datetime import date

#source: https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv

file_name = "constituents.csv"

df_sp500 = pd.read_csv(file_name)
tickers = df_sp500['Symbol'].tolist()

tickers = [t.replace('.', '-') for t in tickers]

print(f"{len(tickers)} Ticker loaded from CSV.")

sectors_dict = {'Industrials': 'XLI',
 'Health Care': 'XLV',
 'Information Technology' : 'XLK',
 'Utilities' : 'XLU',
 'Financials' : 'XLF',
 'Materials' : 'XLB',
 'Consumer Discretionary' : 'XLY',
 'Real Estate': 'XLRE',
 'Communication Services' :'XLC',
 'Consumer Staples':'XLP',
 'Energy': 'XLE'
}

# mass download via yfinance
print("Downloading quotes from yfinance...")
data = yf.download(tickers, period="1y", group_by='ticker')
print()
try:
    close_prices = data.xs('Close', axis=1, level=0)
except KeyError:
    # Falls 'Close' auf Ebene 1 liegt (je nach yfinance Version/Konfiguration)
    close_prices = data.xs('Close', axis=1, level=1)

results = []

# MA Crossing:
for ticker in tickers:
    if ticker not in close_prices.columns:
        continue
        
    series = close_prices[ticker].dropna()
    
    # Enough data?
    if len(series) < 200:
        print(f"{ticker} skipped, only {len(series)} quotes")
        continue

    # Moving Averages:
    ma20 = series.rolling(window=20).mean()
    ma200 = series.rolling(window=200).mean()
    
    # Compare: today vs. 10 days ago
    # .iloc[-1] ist heute, .iloc[-11] ist vor 10 Tagen
    ma20_now = ma20.iloc[-1]
    ma200_now = ma200.iloc[-1]
    ma20_10d_ago = ma20.iloc[-11]
    ma200_10d_ago = ma200.iloc[-11]
    
    # Crossover: 
    if (ma20_10d_ago < ma200_10d_ago) and (ma20_now > ma200_now):
        results.append(ticker)

# Output
print("\nShares with EMA Cross (MA20 crosses MA200 from below in the last 10 days):")
print("Symbol - Sector")
if results:
    for res in results:
        sector = str(df_sp500[df_sp500['Symbol']==res]['GICS Sector'].item())
        print(f"{res} - {sector}")
else:
    print("No hits found.")

# find strong sectors
LOOKBACK = "24mo"
RS_SMOOTH = 40
MOM_WINDOW = 90
MOM_MA = 20
RISK_MA = 200

TRAIL_DAYS = 40 # Trail length in days
TRAIL_POINTS = 4   # Number of segments in the Trail

sector_etfs = {
    # US Sectors:
    "XLC": "Communication Services", #IYZ
    "XLK": "Technology",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials", # Rüstung
    "XLB": "Materials",
    "XLE": "Energy",  # Atomkraft
    "XLV": "Healthcare",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",  # Bergbau
    "XLRE": "Real Estate"
}

# just for info:
#cyclical = {"XLK", "XLF", "XLY", "XLI", "XLB", "XLE"}
#defensive = {"XLV", "XLP", "XLU"}

benchmark = "SPY"
tickers = list(sector_etfs.keys()) + [benchmark]

prices = yf.download(
    tickers,
    period=LOOKBACK,
    interval="1d",
    auto_adjust=False,
    progress=False
)["Adj Close"].dropna()

price_ma = prices.rolling(RS_SMOOTH).mean()

rs = (prices / price_ma).div(
    prices[benchmark] / price_ma[benchmark],
    axis=0
)

momentum = rs.pct_change(MOM_WINDOW)
momentum_ma = momentum.rolling(MOM_MA).mean()

rs_now = rs.iloc[-1]
mom_now = momentum.iloc[-1]
mom_ma_now = momentum_ma.iloc[-1]

spy = prices[benchmark]

ranking = pd.DataFrame({
    "RS": rs_now,
    "Momentum_MA": mom_ma_now
}).drop(index=benchmark)

ranking.index.name = "Ticker"   # <<< WICHTIG

ranking["RS_z"] = (ranking["RS"] - ranking["RS"].mean()) / ranking["RS"].std()
ranking["Mom_z"] = (ranking["Momentum_MA"] - ranking["Momentum_MA"].mean()) / ranking["Momentum_MA"].std()
ranking["Score"] = ranking["RS_z"] + ranking["Mom_z"]
ranking["Sector"] = ranking.index.map(sector_etfs)
ranking["myScore"] = ranking["Momentum_MA"] * (rs_now - 1.0)
ranking["mom_now"] = mom_now
ranking["rs_now"] = rs_now
ranking.loc[(ranking["Momentum_MA"] < 0) & (ranking["rs_now"] < 1), "myScore"] = -1

ranking = ranking.sort_values("myScore", ascending=False)

#print("\nSector ranking:")
#print(ranking)

positiv_sectors = ranking[ranking['myScore'] > 0]
positiv_secors_list = list(positiv_sectors['Sector'].index)
print("Sectors in the upper right quadrant:")
print(positiv_secors_list)

# filter against strong sectors
print("\nMA crossings: shares in strong sectors:")
results_in_strong_sectors = []
if results:
    for res in results:
        sector = str(df_sp500[df_sp500['Symbol']==res]['GICS Sector'].item())
        if sectors_dict[sector] in positiv_secors_list:
            results_in_strong_sectors.append(res)
            print(f"{res} - {sector}")
else:
    print("No hits found.")

print("\nNow checking for seasonality...")

def start_end_for_seaso():
    today = date.today()
    last_year_last_date = date(today.year - 1, 12, 31)
    year_15_before = today.year - 15
    jan1_15_before = date(year_15_before, 1, 1)

    start = jan1_15_before.strftime("%Y-%m-%d")
    end = last_year_last_date.strftime("%Y-%m-%d")
    return start, end

def fit_prophet(df, verbose=False):
    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    print("----- Fitting Prophet model ------") if verbose else None
    m.fit(df)
    print("----------- Model fitted. ------------") if verbose else None
    future = m.make_future_dataframe(periods=365) 
    forecast = m.predict(future)
    return m, forecast

def create_df_for_seaso(quotes15y):
    # nach diesem Schema df erzeugen mit ds und y
    #df_for_seaso = yf.download(ticker,  start=start, end=end, progress=True)
    if quotes15y.empty:
        raise RuntimeError(f"No data for {ticker}")
    # handle multiindex columns (yfinance sometimes returns these)
    if isinstance(quotes15y.columns, pd.MultiIndex):
        quotes15y.columns = quotes15y.columns.droplevel(1)
    price_col = "Adj Close" if "Adj Close" in quotes15y.columns else "Close"
    price_col = "Close"
    s = quotes15y[price_col].dropna()
    # use log-price to make additive seasonal effects easier to interpret (optional)
    #s = np.log(s)
    s.name = "y"
    df_for_seaso = pd.DataFrame({"ds": s.index, "y": s.values})
    return df_for_seaso

def get_fc_min_max_dates(forecast):
    fc_idx_min = forecast.yearly.idxmin()
    fc_idx_max = forecast.yearly.idxmax()
    fc_min_date = forecast.ds.iloc[fc_idx_min] 
    fc_max_date = forecast.ds.iloc[fc_idx_max]
    return fc_min_date, fc_max_date

def today_in_period(fc_min_date, fc_max_date, today=None):
    today = date.today()
    # normalize to date
    fc_min_date = fc_min_date.date()
    fc_max_date = fc_max_date.date()

    start = date(today.year, fc_min_date.month, fc_min_date.day)
    end = date(today.year, fc_max_date.month, fc_max_date.day)

    if start > end:
        start = date(today.year - 1, fc_min_date.month, fc_min_date.day)

    return start <= today <= end, start, end

import logging
# Suppress INFO-Logs from cmdstanpy and prophet logging
logging.getLogger("fbprophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled=True

start_date, end_date = start_end_for_seaso()

for ticker in results_in_strong_sectors:
    quotes15y = yf.download(ticker,  start=start_date, end=end_date, progress=False)
    df_for_seaso = create_df_for_seaso(quotes15y)
    m, forecast = fit_prophet(df_for_seaso, verbose=False)
    fc_min_date, fc_max_date = get_fc_min_max_dates(forecast)
    is_in_period, start, end = today_in_period(fc_min_date, fc_max_date)
    print(f"Seaso for {ticker} from {start} to {end}. Today in seaso period? {is_in_period}")