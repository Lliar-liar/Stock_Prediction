# ModelEnsemble.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

#%%
class ModelComparison:
    def __init__(self, X_train, y_train, X_val, y_val):
        """
        模型对比类
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_val_scaled = self.scaler.transform(X_val)
        
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """
        初始化多个模型进行对比
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            ),
            
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }
        
    def train_and_evaluate_models(self):
        """
        训练和评估所有模型
        """
        # 在方法开始添加类别权重计算
        from sklearn.utils.class_weight import compute_class_weight
    
        # 计算类别权重以处理不平衡数据
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
        class_weight_dict = dict(zip(classes, class_weights))
    
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
        
        # 为支持class_weight的模型设置权重
            if hasattr(model, 'class_weight') and name not in ['XGBoost', 'LightGBM']:
                model.set_params(class_weight=class_weight_dict)
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # 某些模型需要标准化的数据
            if name in ['SVM', 'Neural Network', 'Logistic Regression']:
                X_train_use = self.X_train_scaled
                X_val_use = self.X_val_scaled
            else:
                X_train_use = self.X_train
                X_val_use = self.X_val
            
            # 训练模型
            model.fit(X_train_use, self.y_train)
            
            # 预测
            y_pred = model.predict(X_val_use)
            y_pred_proba = model.predict_proba(X_val_use) if hasattr(model, 'predict_proba') else None
            
            # 评估指标
            accuracy = accuracy_score(self.y_val, y_pred)
            f1_weighted = f1_score(self.y_val, y_pred, average='weighted')
            f1_macro = f1_score(self.y_val, y_pred, average='macro')
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, X_train_use, self.y_train, 
                cv=StratifiedKFold(n_splits=5), 
                scoring='accuracy'
            )
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_val, y_pred),
                'classification_report': classification_report(self.y_val, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1-weighted: {f1_weighted:.4f}")
            
    def visualize_results(self, save_dir=None):
        """
        可视化模型对比结果
        """
        if save_dir is None:
            save_dir = './'
        # 1. 性能对比条形图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准确率对比
        accuracies = {name: res['accuracy'] for name, res in self.results.items()}
        axes[0, 0].bar(accuracies.keys(), accuracies.values())
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1分数对比
        f1_scores = {name: res['f1_weighted'] for name, res in self.results.items()}
        axes[0, 1].bar(f1_scores.keys(), f1_scores.values())
        axes[0, 1].set_title('Model F1-Score (Weighted) Comparison')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 交叉验证结果
        cv_means = {name: res['cv_mean'] for name, res in self.results.items()}
        cv_stds = {name: res['cv_std'] for name, res in self.results.items()}
        x_pos = np.arange(len(cv_means))
        axes[1, 0].bar(x_pos, list(cv_means.values()), yerr=list(cv_stds.values()), capsize=5)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(list(cv_means.keys()), rotation=45)
        axes[1, 0].set_title('Cross-Validation Scores')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('CV Score')
        
        # 综合性能雷达图
        categories = ['Accuracy', 'F1-Weighted', 'F1-Macro', 'CV-Mean']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        
        for name, result in list(self.results.items())[:3]:  # 只画前3个模型
            values = [
                result['accuracy'],
                result['f1_weighted'],
                result['f1_macro'],
                result['cv_mean']
            ]
            values += values[:1]
            ax.plot(angles, values, marker='o', label=name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 混淆矩阵对比
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            if idx < 8:
                sns.heatmap(
                    result['confusion_matrix'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    ax=axes[idx]
                )
                axes[idx].set_title(f'{name} Confusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()


class EnsembleLearning:
    def __init__(self, base_models_results, X_train, y_train, X_val, y_val):
        """
        集成学习类
        
        Args:
            base_models_results: 基础模型的结果字典
            X_train, y_train, X_val, y_val: 训练和验证数据
        """
        self.base_models_results = base_models_results
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.ensemble_methods = {}
        self.ensemble_results = {}
        
    def create_voting_ensemble(self, voting='soft'):
        """
        创建投票集成模型
        """
        # 选择表现最好的几个模型
        sorted_models = sorted(
            self.base_models_results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )[:5]  # 选择前5个模型
        
        estimators = [(name, result['model']) for name, result in sorted_models]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        # 需要重新训练投票分类器
        print(f"Training {voting} voting ensemble...")
        voting_clf.fit(self.X_train, self.y_train)
        
        self.ensemble_methods[f'{voting}_voting'] = voting_clf
        
        return voting_clf
    
    def create_stacking_ensemble(self):
        """
        创建堆叠集成模型
        """
        from sklearn.ensemble import StackingClassifier
        
        # 选择基础模型
        base_estimators = []
        for name, result in list(self.base_models_results.items())[:5]:
            base_estimators.append((name, result['model']))
        
        # 元学习器
        meta_learner = LogisticRegression(max_iter=1000)
        
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5  # 使用5折交叉验证生成元特征
        )
        
        print("Training stacking ensemble...")
        stacking_clf.fit(self.X_train, self.y_train)
        
        self.ensemble_methods['stacking'] = stacking_clf
        
        return stacking_clf
    
    def create_blending_ensemble(self):
        """
        创建混合集成模型
        """
        # 获取所有基础模型的预测概率
        blend_features_train = []
        blend_features_val = []
        
        for name, result in self.base_models_results.items():
            if result['probabilities'] is not None:
                # 获取验证集预测
                blend_features_val.append(result['probabilities'])
                
                # 需要在训练集上重新预测
                model = result['model']
                if hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(self.X_train)
                    blend_features_train.append(train_proba)
        
        # 构建混合特征
        blend_train = np.hstack(blend_features_train)
        blend_val = np.hstack(blend_features_val)
        
        # 训练元模型
        meta_model = GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        )
        meta_model.fit(blend_train, self.y_train)
        
        # 保存混合信息
        self.ensemble_methods['blending'] = {
            'meta_model': meta_model,
            'base_models': self.base_models_results
        }
        
        return meta_model
    
    def evaluate_ensemble_methods(self):
        """
        评估所有集成方法
        """
        for name, ensemble in self.ensemble_methods.items():
            print(f"\nEvaluating {name} ensemble...")
            
            if name == 'blending':
                # 混合方法需要特殊处理
                blend_features = []
                for model_name, result in self.base_models_results.items():
                    if result['probabilities'] is not None:
                        blend_features.append(result['probabilities'])
                
                X_val_blend = np.hstack(blend_features)
                y_pred = ensemble['meta_model'].predict(X_val_blend)
                y_pred_proba = ensemble['meta_model'].predict_proba(X_val_blend)
            else:
                # 其他集成方法
                y_pred = ensemble.predict(self.X_val)
                y_pred_proba = ensemble.predict_proba(self.X_val) if hasattr(ensemble, 'predict_proba') else None
            
            # 计算评估指标
            accuracy = accuracy_score(self.y_val, y_pred)
            f1_weighted = f1_score(self.y_val, y_pred, average='weighted')
            f1_macro = f1_score(self.y_val, y_pred, average='macro')
            
            self.ensemble_results[name] = {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_val, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1-weighted: {f1_weighted:.4f}")
    
    def compare_with_base_models(self):
        """
        比较集成模型与基础模型的性能
        """
        # 收集所有结果
        all_results = {}
        
        # 基础模型结果
        for name, result in self.base_models_results.items():
            all_results[f'Base_{name}'] = {
                'accuracy': result['accuracy'],
                'f1_weighted': result['f1_weighted']
            }
        
        # 集成模型结果
        for name, result in self.ensemble_results.items():
            all_results[f'Ensemble_{name}'] = {
                'accuracy': result['accuracy'],
                'f1_weighted': result['f1_weighted']
            }
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按准确率排序
        sorted_by_accuracy = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # 准确率对比
        names = [item[0] for item in sorted_by_accuracy]
        accuracies = [item[1]['accuracy'] for item in sorted_by_accuracy]
        colors = ['green' if 'Ensemble' in name else 'blue' for name in names]
        
        ax1.barh(names, accuracies, color=colors)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xlim(0, 1)
        
        # 添加数值标签
        for i, v in enumerate(accuracies):
            ax1.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # F1分数对比
        f1_scores = [all_results[name]['f1_weighted'] for name in names]
        
        ax2.barh(names, f1_scores, color=colors)
        ax2.set_xlabel('F1-Score (Weighted)')
        ax2.set_title('F1-Score Comparison')
        ax2.set_xlim(0, 1)
        
        # 添加数值标签
        for i, v in enumerate(f1_scores):
            ax2.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Base Models'),
            Patch(facecolor='green', label='Ensemble Models')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_best_model(self, save_path='best_model.pkl', save_dir=None):
        """
        保存最佳模型
        """
        if save_dir is not None:
            save_path = os.path.join(save_dir, save_path)
        # 找出最佳模型
        all_models = {}
        
        for name, result in self.base_models_results.items():
            all_models[f'base_{name}'] = {
                'model': result['model'],
                'accuracy': result['accuracy']
            }
        
        for name, result in self.ensemble_results.items():
            all_models[f'ensemble_{name}'] = {
                'model': self.ensemble_methods[name],
                'accuracy': result['accuracy']
            }
        
        best_model_name = max(all_models.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = all_models[best_model_name]['model']
        
        scaler_to_save = None
        if hasattr(self, 'scaler'):
            scaler_to_save = self.scaler
        elif hasattr(self, 'model_comparison') and hasattr(self.model_comparison, 'scaler'):
            scaler_to_save = self.model_comparison.scaler
        
        # 保存模型
        joblib.dump({
            'model': best_model,
            'model_name': best_model_name,
            'accuracy': all_models[best_model_name]['accuracy'],
            'scaler': self.scaler if hasattr(self, 'scaler') else None
        }, save_path)
        
        print(f"Best model saved: {best_model_name} (Accuracy: {all_models[best_model_name]['accuracy']:.4f})")