# --- START OF FILE main.py ---

import pandas as pd
import datetime
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from ImportFinancialData import FinancialDataImporter
from LabelFinancialData import FinancialDataLabeler
from ClassifyFinancialNews import FinancialNewsClassifier 
from SimulatePortfolio import PortfolioSimulator

pd.options.mode.chained_assignment = None


class FinancialNewsPredictor:
    def __init__(self, data_source, base_directory='./', 
                 price_csv_path=None, # ADDED: Path to the price CSV
                 selection=None, selection_mode=None, 
                 default_ticker='AAPL', 
                 time_col='Time', headline_col='Headlines',
                 api_call_delay=1):
        
        self.directory_main = Path(base_directory)
        self.price_csv_path = price_csv_path # STORED
        # self.api_call_delay = api_call_delay # Potentially not needed if not using yfinance for anything

        if self.price_csv_path is None:
            # If you want to make it mandatory, raise error. Otherwise, it might fall back to yfinance (if not fully removed)
            # For this exercise, we assume it's provided if yfinance is to be avoided.
            print("Warning: price_csv_path is not provided. Price import will likely fail if yfinance is disabled.")


        if isinstance(data_source, pd.DataFrame):
            self.data = data_source.copy() # Work on a copy
            if 'id' not in self.data.columns:
                 self.data['id'] = range(len(self.data))
            if 'ticker' not in self.data.columns and default_ticker:
                print(f"Warning: 'ticker' column not found in DataFrame. Using default_ticker: {default_ticker}")
                self.data['ticker'] = default_ticker
            elif 'ticker' not in self.data.columns and not default_ticker:
                raise ValueError("'ticker' column missing and no default_ticker provided.")

            if 'release_date' not in self.data.columns or 'content' not in self.data.columns:
                raise ValueError("Input DataFrame must contain 'release_date' and 'content' columns.")
            if not pd.api.types.is_datetime64_any_dtype(self.data['release_date']):
                 self.data['release_date'] = pd.to_datetime(self.data['release_date'], errors='coerce')

        elif isinstance(data_source, (str, Path)): 
            try:
                self.data = pd.read_csv(data_source)
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found at {data_source}")

            if time_col not in self.data.columns or headline_col not in self.data.columns:
                raise ValueError(f"CSV must contain '{time_col}' and '{headline_col}' columns.")
            
            self.data.rename(columns={time_col: 'release_date', headline_col: 'content'}, inplace=True)
            
            def parse_date_flexible(date_str_val):
                date_str = str(date_str_val) # Ensure it's a string
                if pd.isna(date_str_val) or date_str.lower() == 'nan': return pd.NaT

                for fmt in ("%d-%b-%y", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%b-%Y"):
                    try:
                        dt_obj = datetime.datetime.strptime(date_str, fmt)
                        if (fmt == "%d-%b-%y" or fmt == "%d-%b-%Y") and dt_obj.year > datetime.datetime.now().year + 5 : # Heuristic for 2-digit year
                             dt_obj = dt_obj.replace(year=dt_obj.year - 100)
                        return pd.to_datetime(dt_obj)
                    except (ValueError, TypeError):
                        continue
                try: 
                    return pd.to_datetime(date_str)
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse date string: {date_str}")
                    return pd.NaT

            self.data['release_date'] = self.data['release_date'].apply(parse_date_flexible)
            self.data.dropna(subset=['release_date'], inplace=True)

            self.data['id'] = range(len(self.data))
            if default_ticker:
                self.data['ticker'] = default_ticker
                print(f"Assigned default_ticker '{default_ticker}' to all news items.")
            elif 'ticker' not in self.data.columns:
                 raise ValueError("'ticker' column missing from CSV and no default_ticker provided.")
        else:
            raise TypeError("data_source must be a pandas DataFrame or a path to a CSV file.")

        self.data.sort_values(by='release_date', inplace=True)

        self.selection = [selection] if isinstance(selection, str) else selection
        self.selection_mode = selection_mode if (selection_mode in ['sector', 'industry', 'ticker'] and self.selection) else None
        
        if not pd.api.types.is_datetime64_any_dtype(self.data['release_date'].dtype): # Should be handled above
             self.data['release_date'] = pd.to_datetime(self.data['release_date'], errors='coerce')
        
        self.ticker_dates = self.data[['id', 'ticker', 'release_date']].copy()
        self.default_ticker = default_ticker

        self.deltas = None
        self.market_data = None
        self.data_importer = None
        self.data_labeler = None
        self.classifier = None 
        self.predictions = None 
        self.directory_selected, self.directory_labeled, self.directory_model = None, None, None
        self.directory_portfolio, self.simulator, self.simulated_portfolio = None, None, None


    def import_financial_data(self, deltas=None):
        if deltas is None:
            deltas = [1, 2, 3, 4, 5, 6, 7, 10, 14] 
        self.deltas = deltas

        deltas_str = ','.join([str(d) for d in self.deltas])
        self.dir_with_deltas = self.directory_main / f'deltas={deltas_str}' 
        self.dir_with_deltas.mkdir(parents=True, exist_ok=True)

        market_data_path = self.dir_with_deltas / 'market_data.csv'
        if market_data_path.is_file():
            print(f"Loading market data from {market_data_path}")
            self.market_data = pd.read_csv(market_data_path, sep='|')
            if 'Unnamed: 0' in self.market_data.columns:
                try:
                    self.market_data.set_index('Unnamed: 0', inplace=True, drop=True)
                except Exception: 
                    self.market_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

            date_cols_to_convert = [col for col in self.market_data.columns if 'date_' in col]
            for col in date_cols_to_convert:
                self.market_data[col] = pd.to_datetime(self.market_data[col], errors='coerce', utc=True)
        else:
            if self.ticker_dates.empty:
                print("ticker_dates is empty. Cannot initialize FinancialDataImporter.")
                self.market_data = pd.DataFrame() 
                return
            
            if self.price_csv_path is None:
                raise ValueError("price_csv_path must be provided to FinancialNewsPredictor if not loading market_data.csv")

            self.data_importer = FinancialDataImporter(
                self.ticker_dates,
                price_csv_path=self.price_csv_path, # PASS THE PATH
                deltas=self.deltas
                # api_call_delay=self.api_call_delay # Not needed by new importer
            )
            self.data_importer.download_prices() # This now reads from CSV
            self.data_importer.insert_prices()
            self.market_data = self.data_importer.df.copy()
            self.market_data.to_csv(market_data_path, sep='|', index=False) 
        
        if self.market_data is not None and not self.market_data.empty and \
           'date_base' in self.market_data.columns and \
           pd.api.types.is_datetime64_any_dtype(self.market_data['date_base']) and \
           self.market_data['date_base'].dt.tz is not None:
            
            if self.data['release_date'].dt.tz is None:
                self.data['release_date'] = self.data['release_date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
            else:
                self.data['release_date'] = self.data['release_date'].dt.tz_convert('UTC')
    
    def apply_selection(self):
        if self.market_data is None or self.market_data.empty:
            print("Market data not imported/loaded or is empty. Skipping selection.")
            if self.dir_with_deltas: 
                 self.directory_selected = self.dir_with_deltas / 'All_no_market_data'
                 self.directory_selected.mkdir(parents=True, exist_ok=True)
            # Initialize market_data_selected as an empty DataFrame if market_data is None or empty
            self.market_data_selected = pd.DataFrame()
            return

        if self.selection is not None and self.selection_mode is not None:
            selection_str = ','.join([str(s) for s in self.selection]) 
            self.directory_selected = self.dir_with_deltas / f'{self.selection_mode}=[{selection_str}]'
            
            if self.selection_mode not in self.market_data.columns:
                print(f"Warning: Selection mode '{self.selection_mode}' not found in market_data columns. Using all market data.")
                self.market_data_selected = self.market_data.copy() 
                self.directory_selected = self.dir_with_deltas / f'All_due_to_missing_{self.selection_mode}_column'
            else:
                # Ensure self.selection items are strings for isin comparison, as column might be object type
                self.market_data_selected = self.market_data[self.market_data[self.selection_mode].astype(str).isin([str(s) for s in self.selection])].copy()
        else:
            self.directory_selected = self.dir_with_deltas / 'All'
            self.market_data_selected = self.market_data.copy() 

        self.directory_selected.mkdir(parents=True, exist_ok=True)
        
        if self.market_data_selected.empty:
             print(f"Warning: market_data_selected is empty after applying selection: {self.selection_mode}={self.selection}")


    def label_financial_data(self, method='single', delta_to_examine=3, threshold=0.1, base_days=None):
        if not hasattr(self, 'directory_selected') or self.directory_selected is None:
            print("Selection not applied yet. Attempting to apply selection first.")
            self.apply_selection()
            if not hasattr(self, 'directory_selected') or self.directory_selected is None:
                print("Cannot label data without a selected directory path. Ensure apply_selection() runs successfully.")
                self.data_for_classifier = pd.DataFrame() 
                self.data = pd.DataFrame() # Ensure self.data is also empty
                return
        
        current_base_days_for_labeler = base_days
        base_days_str_part = ""
        if method == 'MA':
            if base_days is None: 
                 current_base_days_for_labeler = [1, 2, 3, 4, 5, 6, 7] 
            elif isinstance(base_days, int): 
                 current_base_days_for_labeler = [base_days]
            
            if current_base_days_for_labeler:
                base_days_str_part = f",base_days={','.join(map(str, current_base_days_for_labeler))}"

        _delta_to_examine_list = [delta_to_examine] if isinstance(delta_to_examine, int) else delta_to_examine
        delta_examine_str = ','.join(map(str, _delta_to_examine_list))

        self.directory_labeled = self.directory_selected / f'method={method},delta_examine={delta_examine_str},th={threshold}{base_days_str_part}'
        self.directory_labeled.mkdir(parents=True, exist_ok=True)
        labeled_data_path = self.directory_labeled / 'labeled_data.csv'
        
        if self.market_data_selected is None or self.market_data_selected.empty:
            print("No market data available for labeling (self.market_data_selected is empty/None).")
            self.data_for_classifier = pd.DataFrame()
            self.data = pd.DataFrame() # Update self.data
            return
        
        if 'id' not in self.market_data_selected.columns:
            print("Critical 'id' column missing from market_data_selected. Cannot proceed with labeling.")
            self.data_for_classifier = pd.DataFrame()
            self.data = pd.DataFrame() # Update self.data
            return

        if labeled_data_path.is_file():
            print(f"Loading labeled data from {labeled_data_path}")
            self.data_for_classifier = pd.read_csv(labeled_data_path, sep='|')
            if 'Unnamed: 0' in self.data_for_classifier.columns: 
                try:
                    self.data_for_classifier.set_index('Unnamed: 0', inplace=True, drop=True)
                except: self.data_for_classifier.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

            if 'release_date' in self.data_for_classifier.columns:
                self.data_for_classifier['release_date'] = pd.to_datetime(self.data_for_classifier['release_date'], errors='coerce', utc=True)
        else:
            market_data_for_labeling = self.market_data_selected.copy()
            
            required_price_cols = ['open_base']
            if method == 'single':
                for dte_val in _delta_to_examine_list: required_price_cols.append(f'close_+{dte_val}')
            elif method == 'MA' and current_base_days_for_labeler:
                for bd_val in current_base_days_for_labeler: required_price_cols.append(f'close_+{bd_val}')
                for dte_val in _delta_to_examine_list: required_price_cols.append(f'close_+{dte_val}')
            
            existing_required_cols = [col for col in required_price_cols if col in market_data_for_labeling.columns]
            if not existing_required_cols: # If none of the required price cols exist
                 print(f"Critical price columns for labeling (e.g. {required_price_cols}) are missing from market_data_for_labeling. Columns available: {market_data_for_labeling.columns}")
                 self.data_for_classifier = pd.DataFrame()
                 self.data = pd.DataFrame()
                 return

            market_data_for_labeling.dropna(subset=existing_required_cols, how='any', inplace=True)

            if market_data_for_labeling.empty:
                print("Market data for labeling became empty after dropping NaNs in price columns.")
                self.data_for_classifier = pd.DataFrame()
                self.data = pd.DataFrame()
                return

            valid_ids = market_data_for_labeling['id'].unique()
            news_data_for_labeling = self.data[self.data['id'].isin(valid_ids)][['id', 'content', 'release_date', 'ticker']].copy()

            if news_data_for_labeling.empty:
                print("News data for labeling became empty after aligning with valid market data IDs.")
                self.data_for_classifier = pd.DataFrame()
                self.data = pd.DataFrame()
                return

            self.data_labeler = FinancialDataLabeler(
                news_data=news_data_for_labeling, 
                market_data=market_data_for_labeling, 
                deltas=self.deltas, 
                method=method,
                delta_to_examine=_delta_to_examine_list, 
                threshold=threshold,
                base_days=current_base_days_for_labeler
            )
            self.data_labeler.extract_prices_to_compare()
            self.data_labeler.label_and_join()
            self.data_for_classifier = self.data_labeler.news_data.copy()
            # Ensure 'id' is a column for saving, not just index
            self.data_for_classifier.reset_index(drop=True, inplace=True) # if 'id' was index
            if 'id' not in self.data_for_classifier.columns and self.data_for_classifier.index.name == 'id':
                 self.data_for_classifier.reset_index(inplace=True)

            self.data_for_classifier.to_csv(labeled_data_path, sep='|', index=False) # Save without index if id is column

        if not self.data_for_classifier.empty and \
           all(col in self.data_for_classifier.columns for col in ['sell', 'buy', 'do_nothing']):
            print('Class Distribution in Labeled Data (data_for_classifier):')
            s = self.data_for_classifier["sell"].sum()
            b = self.data_for_classifier["buy"].sum()
            d = self.data_for_classifier["do_nothing"].sum()
            t = len(self.data_for_classifier)
            if t == 0: t = 1 
            print(f'  Sell: {s} ({(s * 100 / t):.2f}%)')
            print(f'  Buy: {b} ({(b * 100 / t):.2f}%)')
            print(f'  Do Nothing: {d} ({(d * 100 / t):.2f}%)')
            print(f'  Total: {s + b + d} = {len(self.data_for_classifier)}')
        elif self.data_for_classifier.empty:
            print("Labeled data (data_for_classifier) is empty. No class distribution to show.")
        else:
            print(f"Label columns missing in data_for_classifier (Columns: {self.data_for_classifier.columns}). Skipping class distribution.")
        
        self.data = self.data_for_classifier.copy() # Update self.data


    def create_classifier(self, model_hf_name='distilbert-base-uncased', max_len=256, validation_size=0.2, batch_size=16, split_type='random'):
        if not hasattr(self, 'directory_labeled') or self.directory_labeled is None:
             print("Labeling step not completed (directory_labeled not set). Attempting to label first.")
             self.label_financial_data() 
             if not hasattr(self, 'directory_labeled') or self.directory_labeled is None:
                 print("Cannot create classifier: Labeling failed.")
                 return
        
        if self.data is None or self.data.empty: 
            print("Labeled data (self.data) is empty or None. Cannot create classifier.")
            return
        
        required_cols_for_classifier = ['id', 'content', 'buy', 'sell', 'do_nothing', 'release_date']
        if not all(col in self.data.columns for col in required_cols_for_classifier):
            print("Labeled data (self.data) is missing required columns. Cannot create classifier.")
            print(f"Missing: {set(required_cols_for_classifier) - set(self.data.columns)}")
            print(f"Available columns: {self.data.columns}")
            return


        safe_model_name = model_hf_name.replace("/", "_") 
        self.directory_model = self.directory_labeled / f'model_pytorch={safe_model_name},max_len={max_len},val_size={validation_size},batch={batch_size},split_type={split_type}'
        self.directory_model.mkdir(parents=True, exist_ok=True)

        model_state_path = self.directory_model / 'model.pth'
        predictions_path = self.directory_model / 'predictions.csv'
        
        self.classifier = FinancialNewsClassifier(
            self.data, 
            model_name=model_hf_name, max_len=max_len,
            validation_size=validation_size, batch_size=batch_size,
            split_type=split_type
        )
        self.data = self.classifier.labeled_news.copy() 

        if predictions_path.is_file() and model_state_path.is_file():
            print(f"Loading existing predictions from {predictions_path}")
            self.predictions = pd.read_csv(predictions_path, sep='|')
            # Handle index for predictions
            if 'id' in self.predictions.columns and self.predictions.index.name != 'id':
                 try: self.predictions.set_index('id', inplace=True)
                 except: pass # If 'id' is not unique or other issues
            elif 'Unnamed: 0' in self.predictions.columns and self.predictions['Unnamed: 0'].equals(self.predictions.index.to_series()):
                 self.predictions.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')


            if 'release_date' in self.predictions.columns:
                self.predictions['release_date'] = pd.to_datetime(self.predictions['release_date'], errors='coerce', utc=True)
            print(f"Loading existing model state from {model_state_path}")
            self.classifier.load_model(model_state_path)
        elif model_state_path.is_file():
            print(f"Loading existing model state from {model_state_path}")
            self.classifier.load_model(model_state_path)
            print("Model loaded. Predictions will be generated if needed.")
        else:
            print(f"No existing model found at {model_state_path}. Model needs to be trained.")


    def train_classifier(self, epochs=3, learning_rate=2e-5, early_stopping_patience=2):
        if self.classifier is None:
            print("Classifier not created. Call create_classifier() first.")
            return
        
        model_state_path = self.directory_model / 'model.pth'
        confusion_matrix_path = self.directory_model / 'confusion_matrix.csv'

        if model_state_path.is_file() and (self.predictions is not None and not self.predictions.empty):
            print(f"Model may already be trained (model file exists at {model_state_path} and predictions are loaded). Skipping training.")
            is_model_loaded_in_classifier = any(p.abs().sum() > 0 for p in self.classifier.model.parameters()) if hasattr(self.classifier, 'model') and self.classifier.model is not None else False
            if not is_model_loaded_in_classifier and hasattr(self.classifier, 'load_model'):
                print("Reloading model state into classifier instance.")
                self.classifier.load_model(model_state_path)
            return

        print(f"Training model. Learning rate: {learning_rate}, Epochs: {epochs}")
        self.classifier.train(
            learning_rate=learning_rate, 
            epochs=epochs, 
            early_stopping_patience=early_stopping_patience,
            model_save_path=model_state_path,
            confusion_matrix_save_path=confusion_matrix_path
        )


    def predict_with_classifier(self):
        if self.classifier is None:
            print("Classifier not created. Call create_classifier() first.")
            return
        
        model_state_path = self.directory_model / 'model.pth'
        predictions_path = self.directory_model / 'predictions.csv'

        if predictions_path.is_file():
            print(f"Predictions already exist at {predictions_path}. Loading them.")
            self.predictions = pd.read_csv(predictions_path, sep='|')
            if 'id' in self.predictions.columns and self.predictions.index.name != 'id':
                 try: self.predictions.set_index('id', inplace=True)
                 except: pass
            elif 'Unnamed: 0' in self.predictions.columns and self.predictions['Unnamed: 0'].equals(self.predictions.index.to_series()):
                 self.predictions.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')


            if 'release_date' in self.predictions.columns:
                 self.predictions['release_date'] = pd.to_datetime(self.predictions['release_date'], errors='coerce', utc=True)
            
            if self.classifier.predictions_df is None: 
                 if model_state_path.is_file():
                     self.classifier.load_model(model_state_path) 
                     self.classifier.predict() 
                     self.predictions = self.classifier.predictions_df.copy()
                     # Ensure 'id' is column before saving
                     if self.predictions.index.name == 'id': self.predictions.reset_index(inplace=True)
                     self.predictions.to_csv(predictions_path, sep='|', index=False) 
                 else:
                     print("Model file not found; cannot generate fresh predictions if loaded ones are misaligned.")
            return

        if not model_state_path.is_file():
            print(f"Model not found at {model_state_path}. Train model first.")
            return

        print("Generating predictions with the classifier...")
        self.classifier.load_model(model_state_path) 
        self.classifier.predict() 
        
        self.predictions = self.classifier.predictions_df.copy()
        if self.predictions.index.name == 'id': self.predictions.reset_index(inplace=True)
        self.predictions.to_csv(predictions_path, sep='|', index=False)
        print(f"Predictions saved to {predictions_path}")


    def simulate_portfolio(self, selection=None, starting_amount=100, transaction_amount=1, interactive=False,
                           price='Close', only_validation=False, starting_cash=1e3, start_date=None, end_date=None):
        if self.predictions is None or self.predictions.empty:
            print("Predictions not available or empty. Call predict_with_classifier() first.")
            return
        if self.data is None or self.data.empty: 
            print("Base data (self.data, expected to be labeled) is not available or empty. Cannot run simulation.")
            return
        
        if self.price_csv_path is None:
            raise ValueError("price_csv_path must be provided for PortfolioSimulator when not using yfinance.")


        sim_selection_list = []
        if selection:
            sim_selection_list = [selection] if isinstance(selection, str) else list(selection)
        else: 
            if 'ticker' in self.predictions.columns:
                sim_selection_list = self.predictions['ticker'].unique().tolist()
            elif hasattr(self, 'default_ticker') and self.default_ticker:
                sim_selection_list = [self.default_ticker]
            if not sim_selection_list:
                print("No tickers for simulation (selection is None and no tickers in predictions/default).")
                return
        
        selection_path_str = "AllTickersInPreds" if not selection else ','.join(sim_selection_list)

        self.directory_portfolio = self.directory_model / f'portfolio_sel=[{selection_path_str}],start_stocks={starting_amount},start_cash={starting_cash},trans={transaction_amount}'
        self.directory_portfolio.mkdir(parents=True, exist_ok=True)
        portfolio_csv_path = self.directory_portfolio / 'portfolio.csv'
        
        news_data_for_sim = self.data[['id', 'ticker', 'release_date', 'content']].copy()
        
        # Ensure predictions has 'id' as a column if it's an index
        predictions_for_sim = self.predictions.copy()
        if predictions_for_sim.index.name == 'id':
            predictions_for_sim.reset_index(inplace=True)

        if not portfolio_csv_path.is_file():
            print(f"Simulating portfolio. Saving to: {portfolio_csv_path}")
            self.simulator = PortfolioSimulator(
                news_data=news_data_for_sim, 
                predictions=predictions_for_sim,
                price_csv_path=self.price_csv_path, # PASS THE PATH
                selection=sim_selection_list, 
                starting_amount=starting_amount, transaction_amount=transaction_amount,
                price=price, only_validation=only_validation,
                starting_cash=starting_cash, start_date=start_date, end_date=end_date
                # api_call_delay=self.api_call_delay # Not needed by new simulator
            )
            self.simulator.insert_prices() # Reads from CSV
            self.simulator.insert_quantities()
            self.simulator.compute_total()
            self.simulated_portfolio = self.simulator.portfolio.copy()
            self.simulated_portfolio.to_csv(portfolio_csv_path, index=True) # Portfolio index is Date
        else:
            print(f"Loading existing portfolio simulation from {portfolio_csv_path}")
            self.simulator = PortfolioSimulator( 
                news_data=news_data_for_sim, predictions=predictions_for_sim, 
                price_csv_path=self.price_csv_path, # PASS THE PATH
                selection=sim_selection_list,
                starting_amount=starting_amount, transaction_amount=transaction_amount,
                price=price, only_validation=only_validation,
                starting_cash=starting_cash, start_date=start_date, end_date=end_date
                # api_call_delay=self.api_call_delay
            )
            loaded_portfolio = pd.read_csv(portfolio_csv_path)
            if 'Date' in loaded_portfolio.columns:
                 loaded_portfolio['Date'] = pd.to_datetime(loaded_portfolio['Date'], errors='coerce')
                 if 'Unnamed: 0' in loaded_portfolio.columns and pd.api.types.is_datetime64_any_dtype(loaded_portfolio['Unnamed: 0']):
                     loaded_portfolio.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
                 if 'Date' in loaded_portfolio.columns : 
                    loaded_portfolio.set_index('Date', inplace=True)
            self.simulator.portfolio = loaded_portfolio
            self.simulated_portfolio = self.simulator.portfolio.copy()
            
        self.simulator.visualize(interactive=interactive)
# --- END OF FILE main.py ---