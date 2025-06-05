#%%
#新增调用模块到main.py
import pandas as pd
from DataAnalysisModule import FinancialDataAnalyzer, load_news_data

# 加载新闻数据
news_data = load_news_data('integrated_headlines.csv')

# 加载股价数据
price_data = pd.read_csv('nasdq.csv')  

# 创建分析器实例
analyzer = FinancialDataAnalyzer(news_data, price_data)

# 执行分析
analyzer.analyze_news_sentiment()
analyzer.visualize_sentiment_distribution()
correlation_data, corr_matrix = analyzer.analyze_correlation_with_stock_movement()
analyzer.extract_keywords_and_topics()
analyzer.create_word_cloud()
analyzer.analyze_temporal_patterns()
analyzer.generate_analysis_report()
