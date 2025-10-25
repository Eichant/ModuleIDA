import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from scipy import stats


# Генерація даних (імітація реальних медичних даних)
np.random.seed(42)
n = 300

# Генерація стабілізованої глюкози (нормальний розподіл)
glucose = np.random.normal(6.5, 1.5, n)  # ммоль/л
glucose = np.clip(glucose, 4.0, 12.0)   # обмеження реалістичних значень

# Генерація гемоглобіну з нелінійною залежністю від глюкози
# Використовуємо експоненційну залежність + шум
hemoglobin = 130 + 15 * np.exp(-0.3 * (glucose - 5)) + np.random.normal(0, 8, n)
hemoglobin = np.clip(hemoglobin, 100, 160)  # обмеження реалістичних значень

# Створення DataFrame
df = pd.DataFrame({
    'Глюкоза': glucose,
    'Гемоглобін': hemoglobin
})

print("ОПИС ДАНИХ:")
print(f"   Розмір вибірки: {n} спостережень")
print(f"   Стабілізована глюкоза: M = {df['Глюкоза'].mean():.2f} ± {df['Глюкоза'].std():.2f} ммоль/л")
print(f"   Гемоглобін: M = {df['Гемоглобін'].mean():.2f} ± {df['Гемоглобін'].std():.2f} г/л")

# 2. Візуалізація вихідних даних
plt.figure(figsize=(15, 5))

# Діаграма розсіювання
plt.subplot(1, 3, 1)
plt.scatter(df['Глюкоза'], df['Гемоглобін'], alpha=0.6, color='purple')
plt.xlabel('Стабілізована глюкоза (ммоль/л)')
plt.ylabel('Гемоглобін (г/л)')
plt.title('Діаграма розсіювання\nГлюкоза vs Гемоглобін')
plt.grid(True, alpha=0.3)

# Гістограма глюкози
plt.subplot(1, 3, 2)
plt.hist(df['Глюкоза'], bins=20, color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Глюкоза (ммоль/л)')
plt.ylabel('Частота')
plt.title('Розподіл глюкози')
plt.grid(True, alpha=0.3)

# Гістограма гемоглобіну
plt.subplot(1, 3, 3)
plt.hist(df['Гемоглобін'], bins=20, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Гемоглобін (г/л)')
plt.ylabel('Частота')
plt.title('Розподіл гемоглобіну')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. ВИЗНАЧЕННЯ НЕЛІНІЙНИХ МОДЕЛЕЙ ТА МЕТОДІВ ОЦІНЮВАННЯ

print("\n2. МЕТОДИ ОЦІНЮВАННЯ ПАРАМЕТРІВ НЕЛІНІЙНИХ МОДЕЛЕЙ:")

# Визначення нелінійних моделей
def exponential_model(x, a, b, c):
    """Експоненційна модель: y = a + b * exp(-c*(x-x0))"""
    return a + b * np.exp(-c * (x - 5))

def power_model(x, a, b, c):
    """Степенева модель: y = a + b * x^c"""
    return a + b * np.power(x, c)

def logarithmic_model(x, a, b, c):
    """Логарифмічна модель: y = a + b * ln(x + c)"""
    return a + b * np.log(x + c)

def polynomial_model(x, a, b, c):
    """Поліноміальна модель: y = a + b*x + c*x²"""
    return a + b*x + c*x**2

print("   а) Метод найменших квадратів (Non-linear Least Squares):")
print("      - Мінімізація суми квадратів залишків")
print("      - Використовує алгоритм Левенберга-Марквардта")
print("      - Застосовується для всіх типів нелінійних моделей")

print("   б) Узагальнений метод моментів (GMM):")
print("      - Використання моментних умов для оцінки параметрів")
print("      - Ефективний для моделей з гетероскедастичністю")

print("   в) Метод максимальної правдоподібності (MLE):")
print("      - Максимізація функції правдоподібності")
print("      - Оптимальний для нормально розподілених залишків")

# 4. ОЦІНЮВАННЯ ПАРАМЕТРІВ НЕЛІНІЙНИХ МОДЕЛЕЙ

print("\n3. ОЦІНЮВАННЯ ПАРАМЕТРІВ МОДЕЛЕЙ:")

# Підготовка даних
x_data = df['Глюкоза'].values
y_data = df['Гемоглобін'].values

# Оцінка параметрів для кожної моделі
models = {
    'Експоненційна': (exponential_model, [130, 15, 0.3]),
    'Степенева': (power_model, [130, -5, 0.5]),
    'Логарифмічна': (logarithmic_model, [130, -10, 1]),
    'Поліноміальна': (polynomial_model, [150, -5, 0.5])
}

results = {}

for name, (model, initial_guess) in models.items():
    try:
        # Використання методу найменших квадратів для оцінки параметрів
        params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess, maxfev=5000)
        results[name] = {
            'model': model,
            'params': params,
            'covariance': covariance
        }
        print(f"   {name} модель: параметри = {params}")
    except Exception as e:
        print(f"   {name} модель: помилка оцінювання - {e}")

# 5. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ПІДГОНКИ МОДЕЛЕЙ

plt.figure(figsize=(12, 8))

# Сортування даних для плавних кривих
sort_idx = np.argsort(x_data)
x_sorted = x_data[sort_idx]
y_sorted = y_data[sort_idx]

# Візуалізація всіх моделей
plt.scatter(x_data, y_data, alpha=0.6, color='black', label='Спостереження')

colors = ['red', 'blue', 'green', 'orange']
for i, (name, result) in enumerate(results.items()):
    y_pred = result['model'](x_sorted, *result['params'])
    plt.plot(x_sorted, y_pred, color=colors[i], linewidth=2, label=f'{name} модель')

plt.xlabel('Стабілізована глюкоза (ммоль/л)')
plt.ylabel('Гемоглобін (г/л)')
plt.title('Порівняння нелінійних моделей регресії')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. ОЦІНКА ЯКОСТІ МОДЕЛЕЙ

print("\n4. ОЦІНКА ЯКОСТІ НЕЛІНІЙНИХ МОДЕЛЕЙ:")

quality_metrics = {}

for name, result in results.items():
    y_pred = result['model'](x_data, *result['params'])
    
    # Обчислення метрик якості
    r2 = r2_score(y_data, y_pred)
    mse = mean_squared_error(y_data, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_data - y_pred))
    
    # Стандартні похибки параметрів
    param_errors = np.sqrt(np.diag(result['covariance']))
    
    quality_metrics[name] = {
        'R²': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'params': result['params'],
        'param_errors': param_errors
    }
    
    print(f"\n   {name} модель:")
    print(f"      - R² = {r2:.4f}")
    print(f"      - MSE = {mse:.2f}")
    print(f"      - RMSE = {rmse:.2f}")
    print(f"      - MAE = {mae:.2f}")
    print(f"      - Параметри: {result['params']}")
    print(f"      - Похибки параметрів: {param_errors}")

# Визначення найкращої моделі
best_model = max(quality_metrics.items(), key=lambda x: x[1]['R²'])
print(f"\n   НАЙКРАЩА МОДЕЛЬ: {best_model[0]}")
print(f"      R² = {best_model[1]['R²']:.4f}")

#СТАТИСТИЧНА ПЕРЕВІРКА АДЕКВАТНОСТІ МОДЕЛІ

print("\n5. СТАТИСТИЧНА ПЕРЕВІРКА АДЕКВАТНОСТІ МОДЕЛІ:")

# Аналіз залишків для найкращої моделі
best_model_name = best_model[0]
best_result = results[best_model_name]
y_pred_best = best_result['model'](x_data, *best_result['params'])
residuals = y_data - y_pred_best

# Перевірка нормальності залишків
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"   Тест Шапіро-Вілка на нормальність залишків:")
print(f"      - p-value = {shapiro_p:.4f}")
print(f"      - Залишки {'нормальні' if shapiro_p > 0.05 else 'ненормальні'}")

# Перевірка гомоскедастичності
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_best, residuals, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Прогнозовані значення гемоглобіну (г/л)')
plt.ylabel('Залишки')
plt.title('Діаграма залишків\n(перевірка гомоскедастичності)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Залишки')
plt.ylabel('Частота')
plt.title('Гістограма розподілу залишків')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()