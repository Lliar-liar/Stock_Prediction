# --- START OF FILE ImportFinancialData.py ---

import pandas as pd
# import yfinance as yf # No longer needed for price download
from datetime import timedelta
# import time # No longer needed for yfinance delays

class FinancialDataImporter:
    def __init__(self, ticker_dates, price_csv_path, deltas=None, api_call_delay=1): # Added price_csv_path
        self.df = ticker_dates.copy()
        self.deltas = deltas
        self.price_csv_path = price_csv_path # Store the path
        # self.api_call_delay = api_call_delay # Not strictly needed if not using yf for info

        self.tickers = self.df['ticker'].unique()

        self.df = self.df.rename(columns={'release_date': 'date_base'})
        self.df['date_base'] = pd.to_datetime(self.df['date_base'], utc=True)
        self.df = self.correct_weekends(self.df, date_col='date_base')
        
        if not self.df.empty and 'ticker' in self.df.columns and 'date_base' in self.df.columns:
            self.min_max_dates = self.df \
                .drop(columns=['id'], errors='ignore') \
                .groupby(by=['ticker']) \
                .agg(
                    min_date=pd.NamedAgg(column='date_base', aggfunc='min'),
                    max_date=pd.NamedAgg(column='date_base', aggfunc='max')
                    )
        else:
            self.min_max_dates = pd.DataFrame(columns=['min_date', 'max_date'])

        if self.deltas is not None:
            if isinstance(self.deltas, int):
                self.deltas = [self.deltas]

            for d in self.deltas:
                col_name = 'date_+' + str(d)
                self.df[col_name] = self.df['date_base'] + timedelta(days=d)
                self.df[col_name] = pd.to_datetime(self.df[col_name], utc=True)
                self.df = self.correct_weekends(self.df, date_col=col_name)

                if d == max(self.deltas) and not self.min_max_dates.empty:
                    # Adjust max_date to ensure we have data for the largest delta + buffer
                    self.min_max_dates['max_date'] = self.min_max_dates['max_date'] + timedelta(days=d+2)
        
        elif not self.min_max_dates.empty:
             self.min_max_dates['max_date'] = self.min_max_dates['max_date'] + timedelta(days=2)

        # Define the structure of price_data
        self.price_data = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sector', 'Industry'])
        
        # Load the CSV data once
        try:
            self.market_csv_data = pd.read_csv(self.price_csv_path)
            self.market_csv_data['Date'] = pd.to_datetime(self.market_csv_data['Date'], utc=True)
            # Ensure required columns are present from CSV, can add more error checking
            required_csv_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self.market_csv_data.columns for col in required_csv_cols):
                raise ValueError(f"Price CSV is missing one of required columns: {required_csv_cols}")
        except FileNotFoundError:
            print(f"ERROR: Price CSV file not found at {self.price_csv_path}")
            self.market_csv_data = pd.DataFrame()
        except Exception as e:
            print(f"Error loading price CSV: {e}")
            self.market_csv_data = pd.DataFrame()


    @staticmethod
    def correct_weekends(df, date_col='date'):
        if date_col not in df.columns:
            return df
        df_copy = df.copy()
        df_copy['weekday'] = df_copy[date_col].dt.weekday
        # Monday is 0 and Sunday is 6
        # Move Sunday to Monday
        df_copy.loc[df_copy['weekday'] == 6, date_col] = df_copy.loc[df_copy['weekday'] == 6, date_col] + timedelta(days=1)
        # Move Saturday to Friday
        df_copy.loc[df_copy['weekday'] == 5, date_col] = df_copy.loc[df_copy['weekday'] == 5, date_col] - timedelta(days=1)
        return df_copy.drop(columns=['weekday'])

    def download_prices(self):
        if not hasattr(self, 'min_max_dates') or self.min_max_dates.empty:
            print("min_max_dates not initialized or empty. Cannot process prices from CSV.")
            return
        if self.market_csv_data.empty:
            print("Market CSV data is empty. Cannot process prices.")
            return

        all_price_records = []

        for tick in self.tickers:
            print(f"Processing ticker: {tick} using data from CSV")
            try:
                if tick not in self.min_max_dates.index:
                    print(f"  Ticker {tick} not found in min_max_dates. Skipping.")
                    continue

                start_date_dt = self.min_max_dates.loc[tick, 'min_date']
                end_date_dt = self.min_max_dates.loc[tick, 'max_date']
                
                # Ensure dates are timezone-naive for comparison if market_csv_data['Date'] is naive
                # Or ensure both are timezone-aware (UTC is good)
                # self.market_csv_data['Date'] is already UTC from __init__

                if pd.isna(start_date_dt) or pd.isna(end_date_dt) or start_date_dt >= end_date_dt:
                    print(f"  Skipping {tick} due to invalid/inverted date range: Start {start_date_dt}, End {end_date_dt}")
                    data_hist_for_ticker = pd.DataFrame()
                else:
                    print(f"  Filtering CSV price data for {tick} from {start_date_dt.date()} to {end_date_dt.date()}...")
                    # The CSV data is general, not per-ticker, so we use the same slice for all tickers
                    data_hist_for_ticker = self.market_csv_data[
                        (self.market_csv_data['Date'] >= start_date_dt) &
                        (self.market_csv_data['Date'] <= end_date_dt)
                    ].copy() # Use .copy() to avoid SettingWithCopyWarning

                if data_hist_for_ticker.empty:
                    print(f"  No price data found in CSV for {tick} in the date range.")
                    # Create a placeholder if needed, or just skip
                    current_ticker_df_data = {
                        'Date': [pd.NaT], 'Open': [pd.NA], 'High': [pd.NA], 'Low': [pd.NA], 
                        'Close': [pd.NA], 'Adj Close': [pd.NA], 'Volume': [pd.NA],
                        'Ticker': [tick], 'Sector': ['Unknown (From CSV)'], 
                        'Industry': ['Unknown (From CSV)']
                    }
                    current_ticker_data = pd.DataFrame(current_ticker_df_data)
                else:
                    # Rename columns from CSV to match yfinance output structure
                    # Assuming CSV has 'Open', 'High', 'Low', 'Close', 'Volume'
                    # 'Adj Close' will be same as 'Close'
                    current_ticker_data = data_hist_for_ticker.rename(
                        columns={
                            'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                            'Close': 'Close', 'Volume': 'Volume'
                        }
                    )
                    current_ticker_data['Adj Close'] = current_ticker_data['Close']
                    current_ticker_data['Ticker'] = tick
                    current_ticker_data['Sector'] = 'Unknown (From CSV)'
                    current_ticker_data['Industry'] = 'Unknown (From CSV)'
                    
                expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker', 'Sector', 'Industry']
                for col in expected_cols:
                    if col not in current_ticker_data.columns:
                        current_ticker_data[col] = pd.NaT if col == 'Date' else pd.NA
                
                all_price_records.append(current_ticker_data[expected_cols])

            except Exception as e:
                print(f"An error occurred processing {tick} with CSV data: {e}")
                error_placeholder = pd.DataFrame([{
                    'Date': pd.NaT, 'Ticker': tick, 'Open': pd.NA, 'High': pd.NA, 'Low': pd.NA,
                    'Close': pd.NA, 'Adj Close': pd.NA, 'Volume': pd.NA,
                    'Sector': 'Unknown (Error CSV)', 'Industry': 'Unknown (Error CSV)'
                }])
                all_price_records.append(error_placeholder)
                continue
        
        if all_price_records:
            self.price_data = pd.concat(all_price_records, ignore_index=True)
            if 'Date' in self.price_data.columns: # Already should be UTC
                 self.price_data['Date'] = pd.to_datetime(self.price_data['Date'], utc=True, errors='coerce')
        
        print('Finalized processing of market data from CSV.')


    def insert_prices(self):
        if self.price_data.empty:
            print("Price data is empty. Cannot insert prices into self.df.")
            cols_to_init = ['close_base', 'open_base', 'sector', 'industry']
            if self.deltas:
                cols_to_init.extend([f'close_+{d}' for d in self.deltas])
            for col in cols_to_init:
                if col not in self.df.columns:
                    self.df[col] = pd.NA if 'close' in col or 'open' in col else 'Unknown'
            return
        
        df_to_merge_into = self.df.copy()

        # Prepare base price data for merge
        # Key columns for merging: 'Ticker', 'Date'
        # Price columns to get: 'Close', 'Open'
        # Info columns: 'Sector', 'Industry'
        price_data_base = self.price_data[['Ticker', 'Date', 'Close', 'Open', 'Sector', 'Industry']].copy()
        price_data_base.rename(columns={'Close': 'close_base_temp', 
                                        'Open': 'open_base_temp',
                                        'Sector': 'sector_temp',
                                        'Industry': 'industry_temp'}, inplace=True)
        
        # Ensure date columns are compatible for merging (e.g. UTC, and same type)
        # self.df['date_base'] is already datetime64[ns, UTC]
        # price_data_base['Date'] should also be datetime64[ns, UTC] from download_prices
        if price_data_base['Date'].dt.tz is None and df_to_merge_into['date_base'].dt.tz is not None:
             price_data_base['Date'] = price_data_base['Date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        elif price_data_base['Date'].dt.tz is not None and df_to_merge_into['date_base'].dt.tz is not None:
             price_data_base['Date'] = price_data_base['Date'].dt.tz_convert('UTC')
        
        df_to_merge_into = pd.merge(df_to_merge_into,
                                   price_data_base,
                                   how='left',
                                   left_on=['ticker', 'date_base'],
                                   right_on=['Ticker', 'Date'])

        df_to_merge_into['close_base'] = df_to_merge_into.get('close_base_temp', pd.NA)
        df_to_merge_into['open_base'] = df_to_merge_into.get('open_base_temp', pd.NA)
        df_to_merge_into['sector'] = df_to_merge_into.get('sector_temp', 'Unknown (From CSV)') # Default if merge fails
        df_to_merge_into['industry'] = df_to_merge_into.get('industry_temp', 'Unknown (From CSV)')
        df_to_merge_into.drop(columns=['Ticker', 'Date', 'close_base_temp', 'open_base_temp', 'sector_temp', 'industry_temp'], inplace=True, errors='ignore')


        if self.deltas:
            price_data_delta_src = self.price_data[['Ticker', 'Date', 'Close']].copy()
            # Ensure Date column in price_data_delta_src is timezone-aware (UTC)
            if price_data_delta_src['Date'].dt.tz is None:
                 price_data_delta_src['Date'] = price_data_delta_src['Date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
            else:
                 price_data_delta_src['Date'] = price_data_delta_src['Date'].dt.tz_convert('UTC')

            for d in self.deltas:
                date_plus_d_col = f'date_+{d}'
                close_plus_d_col = f'close_+{d}'
                
                if date_plus_d_col not in df_to_merge_into.columns:
                    print(f"Warning: Delta date column {date_plus_d_col} not found in df. Skipping delta merge for +{d}.")
                    df_to_merge_into[close_plus_d_col] = pd.NA # Ensure column exists
                    continue
                
                # Ensure date columns for merging are compatible
                # df_to_merge_into[date_plus_d_col] is already datetime64[ns, UTC]
                
                price_data_delta_renamed = price_data_delta_src.rename(columns={'Close': close_plus_d_col + '_temp'})
                
                df_to_merge_into = pd.merge(df_to_merge_into,
                                           price_data_delta_renamed,
                                           how='left',
                                           left_on=['ticker', date_plus_d_col],
                                           right_on=['Ticker', 'Date'])
                
                df_to_merge_into[close_plus_d_col] = df_to_merge_into.get(close_plus_d_col + '_temp', pd.NA)
                df_to_merge_into.drop(columns=['Ticker', 'Date', close_plus_d_col + '_temp'], inplace=True, errors='ignore')
        
        self.df = df_to_merge_into
# --- END OF FILE ImportFinancialData.py ---