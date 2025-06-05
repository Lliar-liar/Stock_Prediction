# DataAnalysisModule.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

#根据现有数据集的辅助
def load_news_data(reuters_path='reuters_headlines.csv', integrated_path='integrated_headlines.csv'):
    """
    加载并合并新闻数据集
    
    Args:
        reuters_path: Reuters数据集路径
        integrated_path: 整合数据集路径
    
    Returns:
        合并后的DataFrame
    """
    # 加载Reuters数据
    reuters_df = pd.read_csv(reuters_path)
    reuters_df['source'] = 'reuters'

    integrated_df = pd.read_csv(integrated_path)
    integrated_df['source'] = 'integrated'
    integrated_df['Description'] = ''  #添加空Description列以保持一致
    
    #合并
    combined_df = pd.concat([reuters_df, integrated_df], ignore_index=True)
    
    return combined_df

def check_date_formats(df, date_column='Time'):
    """
    检查数据集中的日期格式
    
    Args:
        df: DataFrame
        date_column: 日期列名
    
    Returns:
        日期格式统计
    """
    date_formats = {}
    
    for date_str in df[date_column].dropna().unique()[:100]:  #检查
        date_str = str(date_str)
        
        #日期格式不同处理
        formats = [
            ('%b %d %Y', 'Mon DD YYYY'),
            ('%b-%y', 'Mon-YY'),
            ('%Y-%m-%d', 'YYYY-MM-DD'),
            ('%d/%m/%Y', 'DD/MM/YYYY'),
            ('%m/%d/%Y', 'MM/DD/YYYY'),
            ('%b-%d', 'Mon-DD')
        ]
        
        for fmt, desc in formats:
            try:
                datetime.strptime(date_str, fmt)
                date_formats[desc] = date_formats.get(desc, 0) + 1
                break
            except:
                continue
    
    return date_formats

class FinancialDataAnalyzer:
    def __init__(self, news_data, price_data):
        """
        初始化数据分析器
        
        Args:
            news_data: 新闻数据DataFrame
            price_data: 股价数据DataFrame (from NASDAQ dataset)
        """
        self.news_data = news_data.copy()
        self.price_data = price_data.copy()
        
        #统一列名映射
        if 'Headlines' in self.news_data.columns:
            self.news_data['content'] = self.news_data['Headlines']
            #如果有Description列，合并到content中
            if 'Description' in self.news_data.columns:
                self.news_data['content'] = self.news_data['Headlines'] + ' ' + self.news_data['Description'].fillna('')
        
        if 'Time' in self.news_data.columns:
            #使用混合格式解析，让pandas自动推断日期格式
            self.news_data['release_date'] = pd.to_datetime(self.news_data['Time'], 
                                                           format='mixed', 
                                                           errors='coerce')
            #新闻日期多样
            #检查是否有解析失败的日期
            failed_dates = self.news_data[self.news_data['release_date'].isna()]
            if len(failed_dates) > 0:
                print(f"Warning: {len(failed_dates)} dates could not be parsed")
                print("Sample failed dates:", failed_dates['Time'].head())
        
        #下载NLTK数据
        try:
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass
        
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_news_sentiment(self):
        """
        使用VADER进行新闻情感分析
        """
        print("Performing sentiment analysis on news headlines...")
        
        #计算情感分数
        sentiments = []
        for content in self.news_data['content']:
            scores = self.sia.polarity_scores(str(content))
            sentiments.append(scores)
        
        #将情感分数添加到DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        self.news_data = pd.concat([self.news_data, sentiment_df], axis=1)
        
        #添加情感类别
        self.news_data['sentiment_category'] = self.news_data['compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        return self.news_data
    
    def visualize_sentiment_distribution(self):
        """
        可视化情感分布
        """
        
        valid_data = self.news_data[self.news_data['release_date'].notna()].copy()
    
        if len(valid_data) == 0:
            print("Error: No valid dates found in the data")
            return
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        #1. 情感类别分布
        sentiment_counts = self.news_data['sentiment_category'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('News Sentiment Distribution')
        
        #情感分数时间序列
        if self.news_data['release_date'].dtype == 'object':
            self.news_data['release_date'] = pd.to_datetime(self.news_data['release_date'], format='%b %d %Y', errors='coerce')
    
        daily_sentiment = self.news_data.groupby(
            self.news_data['release_date'].dt.date
        )['compound'].mean()
    
        
        axes[0, 1].plot(daily_sentiment.index, daily_sentiment.values)
        axes[0, 1].set_title('Average Daily Sentiment Score')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Compound Sentiment Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        #情感分数分布直方图
        axes[1, 0].hist(self.news_data['compound'], bins=50, edgecolor='black')
        axes[1, 0].set_title('Distribution of Compound Sentiment Scores')
        axes[1, 0].set_xlabel('Compound Score')
        axes[1, 0].set_ylabel('Frequency')
        
        #情感分数箱线图
        sentiment_scores = self.news_data[['neg', 'neu', 'pos', 'compound']]
        axes[1, 1].boxplot(sentiment_scores.values, labels=sentiment_scores.columns)
        axes[1, 1].set_title('Sentiment Score Components')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_correlation_with_stock_movement(self):
        """
        分析新闻情感与股价变动的相关性
        """
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
        self.news_data['release_date'] = pd.to_datetime(self.news_data['release_date'])
        
        #计算价格变化
        self.price_data['price_change'] = self.price_data['Close'].pct_change()
        self.price_data['price_direction'] = (self.price_data['price_change'] > 0).astype(int)
        
        #按日期聚合新闻情感
        daily_sentiment = self.news_data.groupby(
            self.news_data['release_date'].dt.date
        ).agg({
            'compound': 'mean',
            'pos': 'mean',
            'neg': 'mean',
            'neu': 'mean'
        }).reset_index()
        daily_sentiment.columns = ['date', 'avg_compound', 'avg_pos', 'avg_neg', 'avg_neu']
        
        #合并价格和情感数据
        price_daily = self.price_data.groupby(
            self.price_data['Date'].dt.date
        ).agg({
            'price_change': 'mean',
            'price_direction': 'mean',
            'Volume': 'sum'
        }).reset_index()
        price_daily.columns = ['date', 'avg_price_change', 'price_up_ratio', 'total_volume']
        
        correlation_data = pd.merge(daily_sentiment, price_daily, on='date', how='inner')
        
        #计算相关性矩阵
        corr_matrix = correlation_data[
            ['avg_compound', 'avg_pos', 'avg_neg', 'avg_price_change', 'price_up_ratio']
        ].corr()
        
        #可视化相关性
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1)
        plt.title('Correlation between News Sentiment and Stock Price Movement')
        plt.tight_layout()
        plt.savefig('sentiment_price_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_data, corr_matrix
    
    def extract_keywords_and_topics(self, n_topics=5, n_words=10):
        """
        提取关键词和主题（LDA）
        """
        #创建词袋模型
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(self.news_data['content'].fillna(''))
        
        # LDA主题建模
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(doc_term_matrix)
        
        #获取主题词
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx]
            })
        
        #可视化
        fig, axes = plt.subplots(1, n_topics, figsize=(20, 5))
        for idx, topic in enumerate(topics):
            axes[idx].barh(topic['words'], topic['weights'])
            axes[idx].set_title(f'Topic {idx + 1}')
            axes[idx].set_xlabel('Weight')
            
        plt.tight_layout()
        plt.savefig('lda_topics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return topics
    
    def create_word_cloud(self, sentiment_filter=None):
        """
        创建词云图
        """
        if sentiment_filter:
            text = ' '.join(
                self.news_data[
                    self.news_data['sentiment_category'] == sentiment_filter
                ]['content'].fillna('')
            )
            title = f'Word Cloud - {sentiment_filter.capitalize()} Sentiment'
        else:
            text = ' '.join(self.news_data['content'].fillna(''))
            title = 'Word Cloud - All News'
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis'
        ).generate(text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20)
        plt.tight_layout()
        plt.savefig(f'wordcloud_{sentiment_filter or "all"}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_temporal_patterns(self):
        """
        分析时间模式（周内、月内模式）
        """
        self.news_data['weekday'] = self.news_data['release_date'].dt.day_name()
        self.news_data['hour'] = self.news_data['release_date'].dt.hour
        self.news_data['month'] = self.news_data['release_date'].dt.month_name()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        #1新闻发布的周内分布
        weekday_counts = self.news_data['weekday'].value_counts()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)
        
        axes[0, 0].bar(weekday_counts.index, weekday_counts.values)
        axes[0, 0].set_title('News Distribution by Weekday')
        axes[0, 0].set_xlabel('Weekday')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        #2情感分数的周内模式
        weekday_sentiment = self.news_data.groupby('weekday')['compound'].mean()
        weekday_sentiment = weekday_sentiment.reindex(weekday_order, fill_value=0)
        
        axes[0, 1].plot(weekday_sentiment.index, weekday_sentiment.values, marker='o')
        axes[0, 1].set_title('Average Sentiment by Weekday')
        axes[0, 1].set_xlabel('Weekday')
        axes[0, 1].set_ylabel('Average Compound Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        #3新闻发布的小时分布
        hour_counts = self.news_data['hour'].value_counts().sort_index()
        
        axes[1, 0].bar(hour_counts.index, hour_counts.values)
        axes[1, 0].set_title('News Distribution by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Count')
        
        #4月度趋势
        monthly_stats = self.news_data.groupby(
            self.news_data['release_date'].dt.to_period('M')
        ).agg({
            'compound': 'mean',
            'content': 'count'
        })
        
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(monthly_stats.index.astype(str), monthly_stats['compound'], 
                        'b-', label='Avg Sentiment')
        ax2.bar(monthly_stats.index.astype(str), monthly_stats['content'], 
                alpha=0.3, color='orange', label='News Count')
        
        axes[1, 1].set_title('Monthly Trends')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Sentiment', color='b')
        ax2.set_ylabel('News Count', color='orange')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analysis_report(self):
        """
        生成分析报告
        """
        report = []
        report.append("=== Financial News Analysis Report ===\n")
        
        #基础统计
        report.append(f"Total news articles: {len(self.news_data)}")
        report.append(f"Date range: {self.news_data['release_date'].min()} to {self.news_data['release_date'].max()}")
        
        #添加数据源统计
        if 'source' in self.news_data.columns:
            source_counts = self.news_data['source'].value_counts()
            report.append("\nData Sources:")
            for source, count in source_counts.items():
                report.append(f"  {source}: {count} articles")
        
        #情感分析统计
        if 'sentiment_category' in self.news_data.columns:
            sentiment_dist = self.news_data['sentiment_category'].value_counts()
            report.append("\nSentiment Distribution:")
            for category, count in sentiment_dist.items():
                report.append(f"  {category}: {count} ({count/len(self.news_data)*100:.1f}%)")
            
            report.append(f"\nAverage sentiment score: {self.news_data['compound'].mean():.3f}")
            report.append(f"Sentiment standard deviation: {self.news_data['compound'].std():.3f}")
        
        #相关性分析
        if hasattr(self, 'correlation_matrix'):
            report.append("\nKey Correlations:")
            report.append(f"  Sentiment vs Price Change: {self.correlation_matrix.loc['avg_compound', 'avg_price_change']:.3f}")
            report.append(f"  Positive sentiment vs Price Up: {self.correlation_matrix.loc['avg_pos', 'price_up_ratio']:.3f}")
        
        #保存报告
        with open('analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report
    
    

