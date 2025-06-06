# --- START OF FILE ClassifyFinancialNews.py ---

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle # For compatibility with original saving idea, though model state_dict is better

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FinancialNewsClassifier:
    def __init__(self, labeled_news, model_name='distilbert-base-uncased', max_len=256, validation_size=0.2,
                 batch_size=16, split_type='random'):
        """
        PyTorch-based financial news classifier.
        - labeled_news: DataFrame with 'id', 'content', 'buy', 'sell', 'do_nothing', 'release_date'
        - model_name: Hugging Face model name.
        - max_len: Max sequence length for tokenizer.
        - ... other params similar to original
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labeled_news = labeled_news.copy()
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.split_type = split_type

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_labels = 3 # buy, sell, do_nothing
        self.label_map = {'buy': 0, 'sell': 1, 'do_nothing': 2}
        self.idx_to_label = {v: k for k, v in self.label_map.items()}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        ).to(self.device)

        self._prepare_data()
        
        self.predictions_df = None # To store predictions similar to original

    def _prepare_data(self):
        # Convert one-hot to single integer labels
        self.labeled_news['label_int'] = self.labeled_news.apply(
            lambda row: self.label_map['buy'] if row['buy'] == 1 else (self.label_map['sell'] if row['sell'] == 1 else self.label_map['do_nothing']),
            axis=1
        )

        if self.split_type == 'random':
            self.df_train, self.df_val = train_test_split(
                self.labeled_news,
                test_size=self.validation_size,
                random_state=42,
                stratify=self.labeled_news['label_int']
            )
        elif self.split_type == 'time_series':
            self.labeled_news = self.labeled_news.sort_values(by=['release_date'])
            train_idx = int(len(self.labeled_news) * (1 - self.validation_size))
            self.df_train = self.labeled_news.iloc[:train_idx]
            self.df_val = self.labeled_news.iloc[train_idx:]
        else:
            raise ValueError("split_type must be 'random' or 'time_series'")

        self.labeled_news['is_validation'] = 0
        self.labeled_news.loc[self.df_val.index, 'is_validation'] = 1
        
        # These are now DataFrames, not lists of texts/labels
        self.X_train_df = self.df_train
        self.y_train_series = self.df_train['label_int']
        self.X_val_df = self.df_val
        self.y_val_series = self.df_val['label_int']

    def _create_data_loader(self, df, labels_series, shuffle=True):
        dataset = NewsDataset(
            texts=df['content'].to_numpy(),
            labels=labels_series.to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=2) # num_workers for faster loading

    def train(self, learning_rate=2e-5, epochs=3, early_stopping_patience=2, model_save_path='./model.pth',
              confusion_matrix_save_path='./confusion_matrix.csv'):
        
        train_data_loader = self._create_data_loader(self.X_train_df, self.y_train_series, shuffle=True)
        val_data_loader = self._create_data_loader(self.X_val_df, self.y_val_series, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=False)
        total_steps = len(train_data_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_val_accuracy = 0
        epochs_no_improve = 0
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_data_loader)
            history['train_loss'].append(avg_train_loss)
            print(f'Train loss: {avg_train_loss}')

            # Validation
            self.model.eval()
            total_val_loss = 0
            val_preds = []
            val_true = []
            with torch.no_grad():
                for batch in val_data_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_data_loader)
            val_accuracy = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average='weighted')
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            print(f'Val loss: {avg_val_loss}, Val Accuracy: {val_accuracy}, Val F1: {val_f1}')

            if val_accuracy > best_val_accuracy:
                print("Validation accuracy improved. Saving model.")
                torch.save(self.model.state_dict(), model_save_path)
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Validation accuracy did not improve. Epochs without improvement: {epochs_no_improve}")

            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break
        
        
        conf_mat_df = pd.DataFrame(
            confusion_matrix(val_true, val_preds, labels=list(self.label_map.values())),
            index=[f'True_{self.idx_to_label[i]}' for i in self.label_map.values()],
            columns=[f'Pred_{self.idx_to_label[i]}' for i in self.label_map.values()]
        )
        conf_mat_df.to_csv(confusion_matrix_save_path)
        print(f"Confusion matrix saved to {confusion_matrix_save_path}")
        
        # Plotting (basic) - could be moved to FinancialNewsPredictor
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(history['train_loss'], label='Train Loss')
        # plt.plot(history['val_loss'], label='Val Loss')
        # plt.legend()
        # plt.title('Loss')
        # plt.subplot(1, 2, 2)
        # plt.plot(history['val_accuracy'], label='Val Accuracy')
        # plt.legend()
        # plt.title('Accuracy')
        # plt.show()


    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self, texts_to_predict=None): # Modified to predict on all data if None
        """
        Predicts labels for a list of texts or for the entire dataset if texts_to_predict is None.
        Populates self.predictions_df
        """
        if texts_to_predict is None:
            # Predict on the entire dataset self.labeled_news
            df_to_predict_on = self.labeled_news
            texts_list = df_to_predict_on['content'].tolist()
        elif isinstance(texts_to_predict, pd.DataFrame):
            df_to_predict_on = texts_to_predict
            texts_list = df_to_predict_on['content'].tolist()
        elif isinstance(texts_to_predict, list):
            df_to_predict_on = pd.DataFrame({'content': texts_to_predict}) # Temporary df
            texts_list = texts_to_predict
        else:
            raise ValueError("texts_to_predict must be a list of strings, a DataFrame, or None")

        self.model.eval()
        predictions = []
        
        # Create a temporary dataset and dataloader for prediction
        # Using dummy labels as they are not used in prediction
        temp_labels = [0] * len(texts_list) 
        predict_dataset = NewsDataset(
            texts=texts_list,
            labels=temp_labels, 
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        predict_dataloader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in predict_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        predicted_labels_str = [self.idx_to_label[p] for p in predictions]

        # Populate self.predictions_df similar to original structure
        # This is the DataFrame that FinancialNewsPredictor expects
        self.predictions_df = self.labeled_news[['id', 'content', 'buy', 'sell', 'do_nothing', 'is_validation']].copy()
        self.predictions_df['prediction'] = predicted_labels_str
        
        return predicted_labels_str 
