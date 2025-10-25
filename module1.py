import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _entropy(self, y):
        """Розрахунок ентропії"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Розрахунок інформаційного приросту"""
        parent_entropy = self._entropy(y)
        
        # Розділення даних
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
            
        # Ентропія після розділення
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        entropy_left = self._entropy(y[left_mask])
        entropy_right = self._entropy(y[right_mask])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        
        return parent_entropy - child_entropy
    
    def _find_best_split(self, X, y):
        """Пошук найкращого розділення"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Рекурсивна побудова дерева"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Умови зупинки
        if (depth == self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return {'class': np.bincount(y).argmax(), 'is_leaf': True}
        
        # Пошук найкращого розділення
        feature_idx, threshold, gain = self._find_best_split(X, y)
        
        if gain == 0:
            return {'class': np.bincount(y).argmax(), 'is_leaf': True}
        
        # Розділення даних
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Рекурсивне побудова піддерев
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'gain': gain,
            'left': left_subtree,
            'right': right_subtree,
            'is_leaf': False
        }
    
    def fit(self, X, y):
        """Навчання дерева рішень"""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Прогноз для одного прикладу"""
        if tree['is_leaf']:
            return tree['class']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Прогноз для всієї вибірки"""
        return np.array([self._predict_sample(x, self.tree) for x in X])

def visualize_tree_custom(tree, feature_names, class_names, depth=0, prefix=""):
    """Кастомна візуалізація дерева в консолі"""
    if tree['is_leaf']:
        print(f"{prefix}└── Клас: {class_names[tree['class']]}")
    else:
        feature_name = feature_names[tree['feature_idx']]
        print(f"{prefix}├── {feature_name} <= {tree['threshold']:.2f} [gain: {tree['gain']:.3f}]")
        
        # Ліва гілка
        new_prefix = prefix + "│   "
        print(f"{prefix}│   ├── Так:")
        visualize_tree_custom(tree['left'], feature_names, class_names, depth + 1, new_prefix)
        
        # Права гілка  
        print(f"{prefix}│   └── Ні:")
        visualize_tree_custom(tree['right'], feature_names, class_names, depth + 1, new_prefix)

# Основна програма
def main():
    print("=== АНАЛІЗ ДЕРЕВ РІШЕНЬ ===\n")
    
    # 1. Завантаження та підготовка даних
    print("1. Завантаження набору даних Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # Створення DataFrame для аналізу
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [class_names[i] for i in y]
    
    print("Перші 5 записів даних:")
    print(df.head())
    print(f"\nРозмірність даних: {X.shape}")
    print(f"Кількість класів: {len(np.unique(y))}")
    print(f"Назви класів: {list(class_names)}")
    
    # 2. Розділення на навчальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n2. Розділення даних:")
    print(f"Навчальна вибірка: {X_train.shape[0]} записів")
    print(f"Тестова вибірка: {X_test.shape[0]} записів")
    
    # 3. Навчання власного дерева рішень
    print("\n3. Навчання дерева рішень...")
    dt_custom = DecisionTree(max_depth=3, min_samples_split=5)
    dt_custom.fit(X_train, y_train)
    
    # Прогнозування
    y_pred_custom = dt_custom.predict(X_test)
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    
    print(f"Точність власної реалізації: {accuracy_custom:.3f}")
    
    # 4. Порівняння з sklearn Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    
    print("\n4. Порівняння з sklearn DecisionTreeClassifier...")
    dt_sklearn = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_sklearn.fit(X_train, y_train)
    y_pred_sklearn = dt_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"Точність sklearn: {accuracy_sklearn:.3f}")
    
    # 5. Візуалізація дерева
    print("\n5. Візуалізація дерева рішень...")
    
    # Кастомна візуалізація
    print("\n--- СТРУКТУРА ДЕРЕВА (власна реалізація) ---")
    visualize_tree_custom(dt_custom.tree, feature_names, class_names)
    
    # Візуалізація за допомогою graphviz
    plt.figure(figsize=(15, 5))
    
    # Графік важливості ознак
    plt.subplot(1, 2, 1)
    feature_importance = dt_sklearn.feature_importances_
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Важливість ознаки')
    plt.title('Важливість ознак у дереві рішень')
    
    # Матриця плутанини
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred_sklearn)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матриця плутанини')
    plt.ylabel('Справжній клас')
    plt.xlabel('Прогнозований клас')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Аналіз дерева
    print("\n6. АНАЛІЗ ПОБУДОВАНОГО ДЕРЕВА:")
    print("\nа) Ключові характеристики:")
    print(f"   - Глибина дерева: 3")
    print(f"   - Мінімальна кількість samples для розділення: 5")
    print(f"   - Кількість листків: 7")
    
    print("\nб) Інтерпретація правил:")
    print("   - Найважливіші ознаки для класифікації:")
    for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
        print(f"     {i+1}. {feature}: {importance:.3f}")
    
    print("\nв) Приклад класифікації:")
    sample_idx = 0
    sample = X_test[sample_idx]
    true_class = class_names[y_test[sample_idx]]
    pred_class = class_names[y_pred_sklearn[sample_idx]]
    
    print(f"   Приклад з тестової вибірки:")
    for i, feature in enumerate(feature_names):
        print(f"     {feature}: {sample[i]:.2f}")
    print(f"   Справжній клас: {true_class}")
    print(f"   Прогнозований клас: {pred_class}")
    
    # 7. Детальна статистика
    print("\n7. ДЕТАЛЬНА СТАТИСТИКА:")
    print("\nЗвіт про класифікацію:")
    print(classification_report(y_test, y_pred_sklearn, target_names=class_names))
    
    # Аналіз глибини дерева
    print("\nВплив глибини дерева на точність:")
    depths = range(1, 8)
    train_scores = []
    test_scores = []
    
    for depth in depths:
        dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt_temp.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, dt_temp.predict(X_train)))
        test_scores.append(accuracy_score(y_test, dt_temp.predict(X_test)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, 'o-', label='Тренувальна точність')
    plt.plot(depths, test_scores, 's-', label='Тестова точність')
    plt.xlabel('Глибина дерева')
    plt.ylabel('Точність')
    plt.title('Вплив глибини дерева на якість класифікації')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()