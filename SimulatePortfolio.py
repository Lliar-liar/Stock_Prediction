# --- START OF FILE SimulatePortfolio.py ---

import pandas as pd
from datetime import timedelta, datetime
from ipywidgets import interact, SelectionSlider, fixed
# import yfinance as yf # No longer needed for price download
import matplotlib.pyplot as plt
import numpy as np
# import time # No longer needed for yfinance delays


class PortfolioSimulator:
    def __init__(self, news_data, predictions, price_csv_path, selection=None, # Added price_csv_path
                 starting_amount=10, starting_cash=1e6,
                 transaction_amount=1, price='Close', only_validation=False, 
                 start_date=None, end_date=None,
                 api_call_delay=1): # api_call_delay can be removed if no yfinance calls at all

        self.transaction = transaction_amount
        self.price_col_name_in_csv = price # Use the 'price' parameter to select column from CSV, e.g. 'Close'
        self.starting_amount = starting_amount
        self.starting_cash = starting_cash
        self.price_csv_path = price_csv_path
        # self.api_call_delay = api_call_delay # Not strictly needed

        if 'content' not in predictions.columns and 'id' in predictions.columns and 'id' in news_data.columns and 'content' in news_data.columns:
            news_content_data = news_data[['id', 'content']].drop_duplicates(subset=['id'])
            self.content_source = pd.merge(predictions[['id']], news_content_data, on='id', how='left')
        elif 'content' in predictions.columns:
            self.content_source = predictions[['id', 'content']].copy()
        else:
            print("PortfolioSimulator: 'content' column is missing. Interactive display might not show content.")
            self.content_source = pd.DataFrame(columns=['id', 'content'])


        if only_validation:
            if 'is_validation' not in predictions.columns:
                raise ValueError("PortfolioSimulator: 'is_validation' column required in predictions for only_validation=True mode.")
            self.predictions_filtered = predictions.loc[predictions['is_validation'] == 1, :][['id', 'prediction']].copy()
        else:
            self.predictions_filtered = predictions[['id', 'prediction']].copy()

        news_data['release_date'] = pd.to_datetime(news_data['release_date'], errors='coerce').dt.tz_localize(None)
        
        self.data = pd.merge(self.predictions_filtered, 
                             news_data[['id', 'ticker', 'release_date']], 
                             on='id', how='left') \
            .sort_values(by=['release_date'])
        
        self.data.dropna(subset=['release_date', 'ticker'], inplace=True)

        _start_date_obj = None
        if start_date is not None:
            _start_date_obj = pd.to_datetime(start_date, errors='coerce').tz_localize(None)
            if pd.notna(_start_date_obj):
                self.data = self.data[self.data['release_date'] >= _start_date_obj]
        else: 
            _start_date_obj = self.data['release_date'].min() if not self.data.empty else pd.Timestamp('today').tz_localize(None) - pd.Timedelta(days=1)


        _end_date_obj = None
        if end_date is not None:
            _end_date_obj = pd.to_datetime(end_date, errors='coerce').tz_localize(None)
            if pd.notna(_end_date_obj):
                 self.data = self.data[self.data['release_date'] <= _end_date_obj]
        else: 
            _end_date_obj = self.data['release_date'].max() if not self.data.empty else pd.Timestamp('today').tz_localize(None)


        if self.data.empty:
            print("PortfolioSimulator: No data available for simulation after date filtering.")
            self._initialize_empty_portfolio()
            return

        if selection is None:
            self.tickers = self.data['ticker'].unique().tolist()
        else:
            self.tickers = [selection] if isinstance(selection, str) else list(selection)
            self.data = self.data[self.data['ticker'].isin(self.tickers)].copy()
            if self.data.empty:
                print(f"PortfolioSimulator: No data for selected tickers: {self.tickers}.")
                self._initialize_empty_portfolio()
                return

        self.quantity_cols = ['Quantity_' + ticker for ticker in self.tickers]
        self.price_cols = ['Price_' + ticker for ticker in self.tickers]
        self.portfolio = pd.DataFrame(columns=['Date', 'Cash', 'Operation'] + self.quantity_cols)
        
        sim_start_date_calc = self.data['release_date'].min() - timedelta(days=1) if not self.data.empty else _start_date_obj - pd.Timedelta(days=1)
        sim_end_date_calc = self.data['release_date'].max() if not self.data.empty else _end_date_obj
        
        if pd.isna(sim_start_date_calc) or pd.isna(sim_end_date_calc) or sim_start_date_calc > sim_end_date_calc :
            print("PortfolioSimulator: Invalid date range for simulation.")
            self._initialize_empty_portfolio()
            return

        self.portfolio['Date'] = pd.date_range(start=sim_start_date_calc, end=sim_end_date_calc)
        self.portfolio['Date'] = self.portfolio['Date'].dt.tz_localize(None) # Ensure naive for merge
        
        self.portfolio.iloc[0, self.portfolio.columns.get_loc('Cash')] = starting_cash
        if self.tickers: 
            for qty_col_name in self.quantity_cols:
                self.portfolio.iloc[0, self.portfolio.columns.get_loc(qty_col_name)] = starting_amount
        self.portfolio.iloc[0, self.portfolio.columns.get_loc('Operation')] = 'do_nothing'

        self.market_benchmark, self.portfolio_benchmark = None, None
        
        # Load CSV data for prices
        try:
            self.market_csv_data = pd.read_csv(self.price_csv_path)
            self.market_csv_data['Date'] = pd.to_datetime(self.market_csv_data['Date']).dt.tz_localize(None) # Naive
            if self.price_col_name_in_csv not in self.market_csv_data.columns:
                raise ValueError(f"Column '{self.price_col_name_in_csv}' not found in price CSV for portfolio simulation.")
        except FileNotFoundError:
            print(f"ERROR: Price CSV file not found at {self.price_csv_path} for PortfolioSimulator.")
            self.market_csv_data = pd.DataFrame()
        except Exception as e:
            print(f"Error loading price CSV in PortfolioSimulator: {e}")
            self.market_csv_data = pd.DataFrame()


    def _initialize_empty_portfolio(self):
        self.tickers = []
        self.quantity_cols = []
        self.price_cols = []
        self.portfolio = pd.DataFrame(columns=['Date', 'Cash', 'Operation'])
        self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'])
        if not self.portfolio.empty and self.portfolio.index.name != 'Date': self.portfolio.set_index('Date', inplace=True)
        self.market_benchmark = pd.Series(dtype=float)
        self.portfolio_benchmark = pd.Series(dtype=float)
        self.market_csv_data = pd.DataFrame() # Ensure this is also empty

    def insert_prices(self):
        if self.portfolio.empty or not self.tickers: 
            print("PortfolioSimulator: Portfolio DataFrame is empty or no tickers, cannot insert prices.")
            return
        if self.market_csv_data.empty:
            print("PortfolioSimulator: Market CSV data is empty, cannot insert prices.")
            for tick in self.tickers:
                self.portfolio['Price_' + tick] = 0.0 # Ensure columns exist
            return

        min_date = self.portfolio['Date'].min()
        max_date = self.portfolio['Date'].max()

        if pd.isna(min_date) or pd.isna(max_date):
            print("PortfolioSimulator: Portfolio date range is invalid for fetching prices from CSV.")
            return
            
        # Filter the CSV data for the simulation period once
        simulation_period_prices = self.market_csv_data[
            (self.market_csv_data['Date'] >= min_date) &
            (self.market_csv_data['Date'] <= max_date)
        ][['Date', self.price_col_name_in_csv]].copy() # Use the specified price column
        
        if simulation_period_prices.empty:
            print(f"  Warning: No price data found in CSV for the simulation period. Filling all Price_ticker columns with NaNs then 0.")
            for tick in self.tickers:
                self.portfolio['Price_' + tick] = 0.0
            return

        for tick in self.tickers:
            price_col_for_portfolio = 'Price_' + tick
            # Use the same price series for all tickers, renaming the column
            ticker_price_data = simulation_period_prices.rename(
                columns={self.price_col_name_in_csv: price_col_for_portfolio}
            )[['Date', price_col_for_portfolio]]
            
            self.portfolio = pd.merge(self.portfolio, ticker_price_data, on='Date', how='left')
            self.portfolio[price_col_for_portfolio] = self.portfolio[price_col_for_portfolio].ffill().bfill()
            
            if self.portfolio[price_col_for_portfolio].isnull().all():
                print(f"  Warning: All prices for {tick} (from CSV) are NaN. Filling with 0.")
                self.portfolio[price_col_for_portfolio].fillna(0, inplace=True)

    def insert_quantities(self):
        if self.portfolio.empty or not self.tickers:
            print("PortfolioSimulator: Portfolio DataFrame is empty or no tickers, cannot insert quantities.")
            return
        
        # self.portfolio['Date'] and self.data['release_date'] are already naive
        
        self.portfolio[self.quantity_cols + ['Cash']] = self.portfolio[self.quantity_cols + ['Cash']].ffill()
        
        for i in range(len(self.portfolio)): # Use index-based iteration for .loc
            port_date = self.portfolio['Date'].iloc[i]

            if i > 0: 
                for qty_col in self.quantity_cols:
                    self.portfolio.loc[i, qty_col] = self.portfolio.loc[i-1, qty_col]
                self.portfolio.loc[i, 'Cash'] = self.portfolio.loc[i-1, 'Cash']
            
            self.portfolio.loc[i, 'Operation'] = "do_nothing"

            operations_on_this_date = self.data[self.data['release_date'] == port_date]
            
            for _, op_row in operations_on_this_date.iterrows():
                this_ticker = op_row['ticker']
                operation_signal = op_row['prediction']
                
                if this_ticker not in self.tickers:
                    continue

                qty_col_name = 'Quantity_' + this_ticker
                price_col_name = 'Price_' + this_ticker
                
                if price_col_name not in self.portfolio.columns:
                    print(f"Warning: Price column {price_col_name} missing in portfolio for date {port_date}. Skipping operation.")
                    continue

                transaction_price = self.portfolio.loc[i, price_col_name]
                
                if pd.isna(transaction_price) or transaction_price == 0:
                    self.portfolio.loc[i, 'Operation'] = operation_signal 
                    print(f"Warning: Transaction price for {this_ticker} on {port_date} is {transaction_price}. Skipping transaction part of operation '{operation_signal}'.")
                    continue

                quantity_change = 0
                current_cash_on_date = self.portfolio.loc[i, 'Cash']
                current_qty_on_date = self.portfolio.loc[i, qty_col_name]

                if operation_signal == 'buy':
                    if current_cash_on_date >= self.transaction * transaction_price:
                        quantity_change = self.transaction
                        current_cash_on_date -= quantity_change * transaction_price
                    # else: Insufficient cash, do nothing with quantities
                elif operation_signal == 'sell':
                    if current_qty_on_date >= self.transaction:
                        quantity_change = -self.transaction
                        current_cash_on_date -= quantity_change * transaction_price # cash increases
                    # else: Insufficient stocks, do nothing with quantities
                
                self.portfolio.loc[i, qty_col_name] = current_qty_on_date + quantity_change
                self.portfolio.loc[i, 'Cash'] = current_cash_on_date
                self.portfolio.loc[i, 'Operation'] = operation_signal # Record the signal
        
        self.portfolio.ffill(inplace=True) 


    def compute_total(self):
        if self.portfolio.empty:
            print("PortfolioSimulator: Portfolio is empty, cannot compute total.")
            self._ensure_portfolio_cols_exist()
            return

        if self.portfolio.index.name != 'Date':
            if 'Date' in self.portfolio.columns:
                self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'])
                self.portfolio.set_index('Date', inplace=True)
            else:
                print("PortfolioSimulator: Date information missing for compute_total.")
                self._ensure_portfolio_cols_exist()
                return

        self._ensure_portfolio_cols_exist(ensure_price_qty_too=True)
        self.portfolio.fillna(0, inplace=True)

        self.portfolio['Total_Invested'] = 0
        if self.tickers: 
            for i, tick in enumerate(self.tickers):
                qty_col = self.quantity_cols[i]
                price_col = self.price_cols[i] # e.g. Price_MSFT
                if qty_col in self.portfolio.columns and price_col in self.portfolio.columns:
                     self.portfolio['Total_Invested'] += self.portfolio[qty_col] * self.portfolio[price_col]

        self.portfolio['Total'] = self.portfolio['Total_Invested'] + self.portfolio['Cash']
        
        if not self.portfolio.empty and self.portfolio['Total'].iloc[0] != 0:
            self.portfolio['Return'] = (self.portfolio['Total'] - self.portfolio['Total'].iloc[0]) * 100 / self.portfolio['Total'].iloc[0]
        else:
            self.portfolio['Return'] = 0.0 

    def _ensure_portfolio_cols_exist(self, ensure_price_qty_too=False):
        if 'Total' not in self.portfolio.columns: self.portfolio['Total'] = np.nan
        if 'Return' not in self.portfolio.columns: self.portfolio['Return'] = np.nan
        if 'Total_Invested' not in self.portfolio.columns: self.portfolio['Total_Invested'] = np.nan
        if 'Cash' not in self.portfolio.columns: self.portfolio['Cash'] = 0.0
        if 'Operation' not in self.portfolio.columns: self.portfolio['Operation'] = "do_nothing"

        if ensure_price_qty_too:
            for p_col in self.price_cols:
                if p_col not in self.portfolio.columns: self.portfolio[p_col] = 0.0
            for q_col in self.quantity_cols:
                if q_col not in self.portfolio.columns: self.portfolio[q_col] = 0.0


    def visualize(self, interactive=False):
        if self.portfolio.empty:
            print("PortfolioSimulator: Cannot visualize empty portfolio.")
            return
            
        if self.portfolio.index.name != 'Date':
            if 'Date' in self.portfolio.columns:
                self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'])
                self.portfolio.set_index('Date', inplace=True)
            elif isinstance(self.portfolio.index, pd.DatetimeIndex):
                 pass 
            else:
                 print("PortfolioSimulator: Date information missing for visualization.")
                 return

        min_date = self.portfolio.index.min()
        max_date = self.portfolio.index.max()

        if pd.isna(min_date) or pd.isna(max_date) or self.market_csv_data.empty:
            self.market_benchmark = pd.Series(dtype=float, index=self.portfolio.index) 
            self.portfolio_benchmark = pd.Series(dtype=float, index=self.portfolio.index) 
            print("Warning: Market benchmark cannot be calculated due to date or CSV data issues.")
        else:
            # Market benchmark (using the provided CSV as the market)
            benchmark_prices = self.market_csv_data[
                (self.market_csv_data['Date'] >= min_date) & 
                (self.market_csv_data['Date'] <= max_date)
            ][['Date', self.price_col_name_in_csv]].set_index('Date')

            if not benchmark_prices.empty and not benchmark_prices[self.price_col_name_in_csv].isnull().all():
                bm_price_series = benchmark_prices[self.price_col_name_in_csv].ffill().bfill()
                if not bm_price_series.empty and bm_price_series.iloc[0] != 0:
                     self.market_benchmark = (bm_price_series - bm_price_series.iloc[0]) * 100 / bm_price_series.iloc[0]
                else: 
                    self.market_benchmark = pd.Series(0.0, index=self.portfolio.index)
                    print(f"Warning: CSV Market benchmark ({self.price_col_name_in_csv}) data issue (empty or starts at 0).")
            else:
                print(f"Warning: CSV Market benchmark ({self.price_col_name_in_csv}) data issue.")
                self.market_benchmark = pd.Series(dtype=float, index=self.portfolio.index)
            
            # Portfolio benchmark (Buy and Hold, prices from CSV)
            if not self.tickers or not self.price_cols: 
                self.portfolio_benchmark = pd.Series(0.0, index=self.portfolio.index) 
            else:
                # Get the first valid price from the CSV for each ticker (all will be same)
                # For this, we'll use the already merged Price_{ticker} columns in self.portfolio
                initial_prices_series = self.portfolio[self.price_cols].iloc[0]
                
                if not initial_prices_series.empty and not (initial_prices_series == 0).all():
                    initial_investment_value = (initial_prices_series * self.starting_amount).sum()
                    initial_total_value = initial_investment_value + self.starting_cash

                    if initial_total_value != 0:
                        # Calculate current value of initially bought stocks using daily prices from CSV
                        # Since all Price_{ticker} columns use the same CSV source, we can simplify
                        # Take one of the price_cols as representative, or average if they could differ (not in this setup)
                        representative_price_col = self.price_cols[0] # e.g. Price_MSFT
                        current_stock_values = self.portfolio[representative_price_col] * self.starting_amount * len(self.tickers)
                        current_total_values = current_stock_values + self.starting_cash
                        self.portfolio_benchmark = (current_total_values - initial_total_value) * 100 / initial_total_value
                    else:
                        self.portfolio_benchmark = pd.Series(0.0, index=self.portfolio.index)
                else:
                    self.portfolio_benchmark = pd.Series(0.0, index=self.portfolio.index)
                    print("Warning: Portfolio benchmark calculation issue (initial prices are zero/empty).")


        if interactive:
            if self.data.empty:
                fig, ax = self.plot_single_frame()
                plt.show()
                return

            dates = self.portfolio.index.dropna().unique() # Ensure unique, sorted dates
            if not dates.empty:
                options = [(date.strftime(' %d %b %Y '), date) for date in dates]
                slider_value = options[0][1]
                selection_range_slider = SelectionSlider(
                    value=slider_value, options=options, description='Select a Date:',
                    style={'description_width': 'initial'}, orientation='horizontal',
                    layout={'width': '750px', 'height': '50px'}
                )
                interact(self.return_selected_date, date=selection_range_slider, portfolio_df=fixed(self.portfolio))
            else:
                fig, ax = self.plot_single_frame()
                plt.show()
                print("No valid dates in portfolio index for interactive slider.")

        else:
            fig, ax = self.plot_single_frame()
            plt.show()

    def plot_single_frame(self, selected_date=None): 
        fig, ax = plt.subplots(figsize=(12, 7)) 

        if 'Return' in self.portfolio.columns and not self.portfolio['Return'].empty:
             ax.plot(self.portfolio.index, self.portfolio['Return'], label='Model-Managed Portfolio', linewidth=2)
        if self.market_benchmark is not None and not self.market_benchmark.empty:
             # Align index with portfolio for plotting if necessary
             aligned_market_benchmark = self.market_benchmark.reindex(self.portfolio.index).ffill().bfill()
             ax.plot(aligned_market_benchmark.index, aligned_market_benchmark, label='Market (from CSV)', linestyle='--')
        if self.portfolio_benchmark is not None and not self.portfolio_benchmark.empty:
             aligned_portfolio_benchmark = self.portfolio_benchmark.reindex(self.portfolio.index).ffill().bfill()
             ax.plot(aligned_portfolio_benchmark.index, aligned_portfolio_benchmark, label='Equally Weighted Portfolio (B&H, CSV prices)', linestyle=':')


        if 'Operation' in self.portfolio.columns and len(self.portfolio) < 10000: 
            sell_ops = self.portfolio[self.portfolio['Operation'] == 'sell']
            buy_ops = self.portfolio[self.portfolio['Operation'] == 'buy']
            if not sell_ops.empty:
                ax.scatter(sell_ops.index, sell_ops['Return'], label='Sell Op', marker="v", s=60, c='red', alpha=0.7, zorder=5)
            if not buy_ops.empty:
                ax.scatter(buy_ops.index, buy_ops['Return'], label='Buy Op', marker="^", s=60, c='green', alpha=0.7, zorder=5)

        if selected_date and 'Return' in self.portfolio and not self.portfolio.empty:
            selected_date_naive = pd.to_datetime(selected_date).tz_localize(None)

            if selected_date_naive in self.portfolio.index:
                 min_y = self.portfolio['Return'].min(skipna=True) if not self.portfolio['Return'].empty else -10
                 max_y = self.portfolio['Return'].max(skipna=True) if not self.portfolio['Return'].empty else 10
                 min_y = min(0, min_y if pd.notna(min_y) else -10)
                 max_y = max(0, max_y if pd.notna(max_y) else 10)
                 if min_y == max_y : min_y -=1; max_y +=1

                 ax.plot([selected_date_naive, selected_date_naive],
                         [min_y, max_y], 
                         linestyle='-.', color='gray', alpha=0.8, label=f'Selected: {selected_date_naive.strftime("%Y-%m-%d")}')


        ax.set_title('Simulated Portfolio Return', fontsize=18)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Return [%]', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        return fig, ax

    def return_selected_date(self, date, portfolio_df): 
        current_date_from_slider = pd.to_datetime(date).tz_localize(None) 
        
        all_dates_series = self.data['release_date'] 

        if all_dates_series.empty:
            fig, ax = self.plot_single_frame(selected_date=current_date_from_slider)
            plt.show()
            print("No operational data to display for selected date.")
            return

        time_diffs = (all_dates_series - current_date_from_slider).abs()
        
        if time_diffs.empty or time_diffs.isnull().all():
            fig, ax = self.plot_single_frame(selected_date=current_date_from_slider)
            plt.show()
            print(f"No news data points to match for selected slider date: {current_date_from_slider.strftime('%Y-%m-%d')}")
            return
            
        closest_index_in_data = time_diffs.idxmin()
        op_row = self.data.loc[closest_index_in_data]
        
        content_id = op_row['id']
        ticker = op_row['ticker']
        prediction = op_row['prediction']
        
        content_df_row = self.content_source[self.content_source['id'] == content_id]
        content_text = content_df_row['content'].values[0] if not content_df_row.empty and 'content' in content_df_row and pd.notna(content_df_row['content'].values[0]) else "Content not found."

        fig, ax = self.plot_single_frame(selected_date=current_date_from_slider) 
        plt.show()

        print(f"\n--- News closest to {current_date_from_slider.strftime('%Y-%m-%d %H:%M')} ---")
        print(f"Actual News Date: {op_row['release_date'].strftime('%Y-%m-%d %H:%M')}")
        print(f"COMPANY: {ticker} - MODEL PREDICTION: {prediction}")
        print(f"--- Headline/Content (ID: {content_id}) ---")
        
        line_width = 100
        max_chars_to_display = 1000 
        display_content = str(content_text)[:max_chars_to_display] 
        if len(str(content_text)) > max_chars_to_display:
            display_content += "..."

        for i in range(0, len(display_content), line_width):
            print(display_content[i:i + line_width])
        print("----------------------------------------------------------------\n")

# --- END OF FILE SimulatePortfolio.py ---