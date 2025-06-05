# run_predictor.py

import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from main import FinancialNewsPredictor

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def run_financial_news_predictor_pipeline():
    print("Starting Financial News Prediction Pipeline...")

    # --- 1. Configuration - USER TO MODIFY THESE PATHS AND PARAMETERS ---
    NEWS_CSV_PATH = str(PROJECT_ROOT / "integrated_headlines.csv") 
    PRICE_CSV_PATH = str(PROJECT_ROOT / "nasdq.csv") 
    BASE_OUTPUT_DIRECTORY = str(PROJECT_ROOT / "pipeline_output_csv_prices") # Changed output dir
    Path(BASE_OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True) 

    DEFAULT_TICKER_FOR_NEWS = 'NASDAQ' # Or 'NASDAQ' if you want to treat news as relating to the index
    TIME_COLUMN_NAME = 'Time'
    HEADLINE_COLUMN_NAME = 'Headlines'
    # API_CALL_DELAY = 2 # No longer strictly needed if yfinance is fully removed

    selection_list = None
    selection_mode_str = None

    deltas_list = [1, 2, 3, 4, 5, 6, 7, 10, 14]

    labeling_method = 'MA'
    labeling_delta_to_examine = [5, 6, 7]
    labeling_threshold = 0.01
    labeling_base_days = [1, 2, 3]

    classifier_model_hf_name = 'microsoft/deberta-v3-base'
    classifier_max_len = 32

    classifier_batch_size = 32
    classifier_validation_size = 0.2
    classifier_split_type = 'time_series'

    training_epochs = 15
    training_learning_rate = 1e-5
    training_early_stopping_patience = 3

    sim_start_date = '2019-01-01' 
    sim_end_date = '2020-01-01'
    # If DEFAULT_TICKER_FOR_NEWS is 'MSFT', sim_selection will be ['MSFT'].
    # The MSFT prices will be sourced from nasdq.csv (i.e. NASDAQ index prices).
    sim_selection = [DEFAULT_TICKER_FOR_NEWS] if DEFAULT_TICKER_FOR_NEWS else None
    sim_starting_amount = 1000
    sim_transaction_amount = 1
    sim_starting_cash = 1000.0 
    sim_price_type = 'Close' # This will be the column name from nasdq.csv used for price
    sim_only_validation = True
    sim_interactive = False
    # --- End of Configuration ---

    print(f"\n--- 2. Initializing FinancialNewsPredictor ---")
    try:
        predictor = FinancialNewsPredictor(
            data_source=NEWS_CSV_PATH,
            base_directory=BASE_OUTPUT_DIRECTORY,
            price_csv_path=PRICE_CSV_PATH, # PASS THE PATH
            default_ticker=DEFAULT_TICKER_FOR_NEWS,
            time_col=TIME_COLUMN_NAME,
            headline_col=HEADLINE_COLUMN_NAME,
            selection=selection_list,
            selection_mode=selection_mode_str
            # api_call_delay=API_CALL_DELAY
        )
        print("FinancialNewsPredictor initialized.")
        if predictor.data is not None and not predictor.data.empty:
            print("Initial data head (first 5 rows of processed news):")
            print(predictor.data.head())
            print("\nInitial data info:")
            predictor.data.info()
        else:
            print("Predictor data is empty or None after initialization. Check CSV path and content.")
            return
    except FileNotFoundError as fnf_error:
        print(f"ERROR: CSV file not found. {fnf_error}")
        return
    except Exception as e:
        print(f"Error initializing FinancialNewsPredictor: {e}")
        return

    print(f"\n--- 3. Importing Financial Data (from CSV) ---")
    try:
        predictor.import_financial_data(deltas=deltas_list)
        if predictor.market_data is not None and not predictor.market_data.empty:
            print("Market data processed from CSV and/or loaded.")
            print("Market data head (first 5 rows):")
            print(predictor.market_data.head())
            print(f"Market data columns: {predictor.market_data.columns}")
        else:
            print("Market data is None or empty after import attempt.")
    except Exception as e:
        print(f"Error importing financial data: {e}")

    print(f"\n--- 4. Applying Data Selection (if any) ---")
    try:
        predictor.apply_selection()
        if hasattr(predictor, 'market_data_selected'):
            if predictor.market_data_selected is not None:
                print("Selection applied to market data.")
                print(f"Market data shape after selection: {predictor.market_data_selected.shape}")
                if not predictor.market_data_selected.empty:
                     print("Market data head after selection (first 5 rows):")
                     print(predictor.market_data_selected.head())
                else:
                     print("Market data is empty after selection. Subsequent labeling might fail.")
            else:
                print("market_data_selected is None after apply_selection.")
        else:
            print("market_data_selected attribute not found after apply_selection.")

    except Exception as e:
        print(f"Error applying selection: {e}")

    print(f"\n--- 5. Labeling Financial Data ---")
    try:
        predictor.label_financial_data(
            method=labeling_method,
            delta_to_examine=labeling_delta_to_examine,
            threshold=labeling_threshold,
            base_days=labeling_base_days
        )
        if predictor.data is not None and not predictor.data.empty:
            print("Financial data labeled.")
            print("Labeled data head (first 5 rows):")
            print(predictor.data.head())
            print(f"Labeled data shape: {predictor.data.shape}")
            print(f"Labeled data columns: {predictor.data.columns}")
        else:
            print("Data is None or empty after labeling attempt. Cannot proceed to classification.")
            return
    except Exception as e:
        print(f"Error labeling financial data: {e}")
        if predictor.data is None or predictor.data.empty:
            print("Cannot proceed without labeled data.")
            return

    print(f"\n--- 6. Creating/Loading Classifier ---")
    try:
        predictor.create_classifier(
            model_hf_name=classifier_model_hf_name,
            max_len=classifier_max_len,
            batch_size=classifier_batch_size,
            validation_size=classifier_validation_size,
            split_type=classifier_split_type
        )
        if predictor.classifier:
            print("Classifier created/loaded.")
        else:
            print("Classifier is None after creation attempt. Check logs for errors in labeling or data prep.")
            return
    except Exception as e:
        print(f"Error creating classifier: {e}")
        return

    print(f"\n--- 7. Training Classifier (if not already trained) ---")
    try:
        predictor.train_classifier(
            epochs=training_epochs,
            learning_rate=training_learning_rate,
            early_stopping_patience=training_early_stopping_patience
        )
        print("Classifier training process completed (or skipped if already trained).")
    except Exception as e:
        print(f"Error training classifier: {e}")

    print(f"\n--- 8. Predicting with Classifier ---")
    try:
        predictor.predict_with_classifier()
        if predictor.predictions is not None and not predictor.predictions.empty:
            print("Predictions generated/loaded.")
            print("Predictions head (first 5 rows):")
            print(predictor.predictions.head())
        else:
            print("Predictions are None or empty after attempt. Cannot proceed to simulation.")
            return
    except Exception as e:
        print(f"Error predicting with classifier: {e}")
        if predictor.predictions is None or predictor.predictions.empty:
            print("Cannot proceed to simulation without predictions.")
            return

    print(f"\n--- 9. Evaluating Classifier (Example Metrics) ---")
    if predictor.predictions is not None and not predictor.predictions.empty:
        # Ensure 'id' is a column for merging/indexing if it's currently an index
        predictions_df_for_eval = predictor.predictions.copy()
        if predictions_df_for_eval.index.name == 'id':
            predictions_df_for_eval.reset_index(inplace=True)

        val_data = predictions_df_for_eval[predictions_df_for_eval['is_validation'] == 1]

        if not val_data.empty and all(col in val_data.columns for col in ['buy', 'sell', 'do_nothing', 'prediction']):
            def get_true_label_str(row):
                if row['buy'] == 1: return 'buy'
                if row['sell'] == 1: return 'sell'
                return 'do_nothing'
            
            y_true_str = val_data.apply(get_true_label_str, axis=1)
            y_pred_str = val_data['prediction']

            if not y_true_str.empty and not y_pred_str.empty:
                try:
                    labels_for_metrics = ['buy', 'sell', 'do_nothing']
                    val_accuracy = accuracy_score(y_true_str, y_pred_str)
                    
                    y_pred_str_filtered = y_pred_str[y_pred_str.isin(labels_for_metrics)]
                    y_true_str_filtered = y_true_str[y_pred_str.isin(labels_for_metrics)] 

                    if not y_pred_str_filtered.empty:
                        val_f1_weighted = f1_score(y_true_str_filtered, y_pred_str_filtered, labels=labels_for_metrics, average='weighted', zero_division=0)
                        val_mcc = matthews_corrcoef(y_true_str_filtered, y_pred_str_filtered)
                        conf_mat = confusion_matrix(y_true_str_filtered, y_pred_str_filtered, labels=labels_for_metrics)
                        conf_mat_df = pd.DataFrame(conf_mat,
                                                   index=[f'True_{l}' for l in labels_for_metrics],
                                                   columns=[f'Pred_{l}' for l in labels_for_metrics])
                        print("\nValidation Confusion Matrix:")
                        print(conf_mat_df)
                    else:
                        val_f1_weighted = 0.0
                        val_mcc = 0.0
                        print("\nNo predictable classes found in y_pred_str after filtering for known labels.")
                    
                    print(f"Validation Set Metrics:")
                    print(f"  Accuracy (on all val data): {val_accuracy:.4f}")
                    print(f"  F1 Score (Weighted, on predictable classes): {val_f1_weighted:.4f}")
                    print(f"  Matthews Corr Coef (on predictable classes): {val_mcc:.4f}")

                except Exception as e_metrics:
                    print(f"Error calculating metrics: {e_metrics}")
            else:
                print("Not enough data in y_true_str or y_pred_str for metrics calculation.")
        else:
            print("No validation data or required columns found in predictions to calculate metrics.")
    else:
        print("No predictions available to calculate metrics.")

    print(f"\n--- 10. Simulating Portfolio ---")
    try:
        predictor.simulate_portfolio(
            starting_amount=sim_starting_amount,
            transaction_amount=sim_transaction_amount,
            start_date=sim_start_date,
            end_date=sim_end_date,
            selection=sim_selection,
            price=sim_price_type,
            only_validation=sim_only_validation,
            starting_cash=sim_starting_cash,
            interactive=sim_interactive
        )
        if predictor.simulated_portfolio is not None and not predictor.simulated_portfolio.empty:
            print("Portfolio simulation completed.")
            print("Simulated portfolio final state (last 5 rows):")
            print(predictor.simulated_portfolio.tail())
        else:
            print("Simulated portfolio is None or empty after attempt.")

    except Exception as e:
        print(f"Error simulating portfolio: {e}")

    print("\nFinancial News Prediction Pipeline Finished.")


if __name__ == "__main__":
    run_financial_news_predictor_pipeline()
    if plt.get_fignums():
        print("\nDisplaying plots...")
        plt.show()
    else:
        print("\nNo plots to display.")
# --- END OF FILE run_predictor.py ---