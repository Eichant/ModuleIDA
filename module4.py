import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ГЕНЕРАЦІЯ ДАНИХ
np.random.seed(42)
n_samples = 150

spelling_errors = np.random.poisson(3, n_samples)
punctuation_errors = np.random.poisson(2, n_samples)
uniqueness = np.random.uniform(60, 95, n_samples)
text_length = np.random.randint(200, 2000, n_samples)

quality_score = (85 - 3 * spelling_errors - 2 * punctuation_errors + 
                 0.2 * uniqueness + 0.01 * text_length + 
                 np.random.normal(0, 5, n_samples))
quality_score = np.clip(quality_score, 0, 100)

data = pd.DataFrame({
    'Spelling_errors': spelling_errors,
    'Punctuation_errors': punctuation_errors,
    'Uniqueness': uniqueness,
    'Text_length': text_length,
    'Quality_score': quality_score
})

# 2. КОРЕЛЯЦІЙНА МАТРИЦЯ
plt.figure(figsize=(8, 6))
correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', 
            center=0, square=True, fmt='.3f',
            cbar_kws={'label': 'Коефіцієнт кореляції'})

plt.title('Матриця кореляцій між параметрами якості тексту', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 3. ПОРІВНЯЛЬНИЙ АНАЛІЗ ВПЛИВУ ФАКТОРІВ
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Вплив помилок на якість - використовуємо англійські назви стовпців
total_errors = data['Spelling_errors'] + data['Punctuation_errors']
scatter = axes[0].scatter(total_errors, data['Quality_score'], 
                         c=data['Uniqueness'], cmap='viridis', alpha=0.7)
axes[0].set_xlabel('Загальна кількість помилок')
axes[0].set_ylabel('Оцінка якості')
axes[0].set_title('Вплив помилок та унікальності на якість тексту')
plt.colorbar(scatter, ax=axes[0], label='Унікальність (%)')
axes[0].grid(True, alpha=0.3)

# Важливість факторів
X = data[['Spelling_errors', 'Punctuation_errors', 'Uniqueness', 'Text_length']]
y = data['Quality_score']

model = LinearRegression()
model.fit(X, y)

feature_importance = pd.DataFrame({
    'Фактор': ['Орфографічні\nпомилки', 'Пунктуаційні\nпомилки', 'Унікальність', 'Довжина\nтексту'],
    'Вплив': np.abs(model.coef_)
}).sort_values('Вплив', ascending=True)

axes[1].barh(feature_importance['Фактор'], feature_importance['Вплив'],
             color=['#e74c3c', '#e67e22', '#2ecc71', '#3498db'])
axes[1].set_xlabel('Абсолютне значення коефіцієнта')
axes[1].set_title('Відносна важливість факторів якості тексту')
for i, v in enumerate(feature_importance['Вплив']):
    axes[1].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# 4. ДІАГНОСТИКА РЕГРЕСІЙНОЇ МОДЕЛІ
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Фактичні vs Прогнозовані значення
y_pred = model.predict(X)
axes[0].scatter(y, y_pred, alpha=0.7, color='navy')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Фактична оцінка якості')
axes[0].set_ylabel('Прогнозована оцінка якості')
axes[0].set_title(f'Фактичні vs Прогнозовані значення\nR² = {r2_score(y, y_pred):.3f}')
axes[0].grid(True, alpha=0.3)

# Залишки моделі
residuals = y - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.7, color='darkgreen')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Прогнозовані значення')
axes[1].set_ylabel('Залишки')
axes[1].set_title('Діаграма залишків моделі')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. МАТЕМАТИЧНА МОДЕЛЬ ТА ВИСНОВКИ
print("\n" + "="*60)
print("РЕЗУЛЬТАТИ РЕГРЕСІЙНОГО АНАЛІЗУ:")
print("="*60)

equation = f"Оцінка_якості = {model.intercept_:.2f} "
equation += f"+ ({model.coef_[0]:.2f} × Орфографічні_помилки) "
equation += f"+ ({model.coef_[1]:.2f} × Пунктуаційні_помилки) "
equation += f"+ ({model.coef_[2]:.2f} × Унікальність) "
equation += f"+ ({model.coef_[3]:.2f} × Довжина_тексту)"

print(f"\n{equation}")

print(f"\nСТАТИСТИЧНІ ПОКАЗНИКИ МОДЕЛІ:")
print(f"▪ R² (коефіцієнт детермінації): {r2_score(y, y_pred):.4f}")
print(f"▪ RMSE (середньоквадратична помилка): {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
print(f"▪ Середня оцінка якості: {y.mean():.2f} ± {y.std():.2f}")

print(f"\nПРАКТИЧНІ ВИСНОВКИ:")
print(f"▪ Найбільший негативний вплив: орфографічні помилки (-{abs(model.coef_[0]):.2f} бали)")
print(f"▪ Найбільший позитивний вплив: унікальність тексту (+{model.coef_[2]:.2f} бали за 1%)")
print(f"▪ Модель пояснює {r2_score(y, y_pred)*100:.1f}% дисперсії оцінки якості")

# 6. ПРИКЛАД ПРОГНОЗУ
print(f"\n" + "="*60)
print("ПРИКЛАД ПРОГНОЗУ ЯКОСТІ ТЕКСТУ:")
print("="*60)

test_cases = [
    [2, 1, 85, 800],   # Якісний текст
    [8, 5, 65, 300],   # Неякісний текст
    [0, 0, 95, 1500]   # Ідеальний текст
]

descriptions = ["Якісний текст", "Неякісний текст", "Ідеальний текст"]

for i, case in enumerate(test_cases):
    prediction = model.predict([case])[0]
    print(f"\n{descriptions[i]}:")
    print(f"▪ Орфографічні помилки: {case[0]}")
    print(f"▪ Пунктуаційні помилки: {case[1]}") 
    print(f"▪ Унікальність: {case[2]}%")
    print(f"▪ Довжина: {case[3]} слів")
    print(f"▪ Прогнозована оцінка: {prediction:.1f}/100")