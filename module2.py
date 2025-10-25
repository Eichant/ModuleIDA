import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import kstest, shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Генерація даних (аналогічно вашій практичній роботі)
np.random.seed(42)
n = 400

# Генерація ліпопротеїна (логнормальний розподіл)
lipoproteins = np.random.lognormal(mean=3.0, sigma=0.5, size=n)

# Генерація гемоглобіну з негативним зв'язком + шум
hemoglobin = 140 - 0.2 * lipoproteins + np.random.normal(0, 5, n)

# Створення DataFrame
df = pd.DataFrame({
    'Ліпопротеїни': lipoproteins,
    'Гемоглобін': hemoglobin
})

print("1. ОПИС ДАНИХ:")
print(f"   Розмір вибірки: {n} спостережень")
print(f"   Ліпопротеїни: M = {df['Ліпопротеїни'].mean():.2f} ± {df['Ліпопротеїни'].std():.2f}")
print(f"   Гемоглобін: M = {df['Гемоглобін'].mean():.2f} ± {df['Гемоглобін'].std():.2f}")

# 2. Перевірка нормальності розподілу
print("\n2. ПЕРЕВІРКА НОРМАЛЬНОСТІ РОЗПОДІЛУ:")
alpha = 0.05

# Тест Колмогорова-Смірнова
stat_ks_lipo, p_val_ks_lipo = kstest(df['Ліпопротеїни'], 'norm', 
                                    args=(df['Ліпопротеїни'].mean(), df['Ліпопротеїни'].std()))
stat_ks_hemo, p_val_ks_hemo = kstest(df['Гемоглобін'], 'norm', 
                                   args=(df['Гемоглобін'].mean(), df['Гемоглобін'].std()))

print("   Тест Колмогорова-Смірнова:")
print(f"   - Ліпопротеїни: p-value = {p_val_ks_lipo:.4f} {'(нормальний)' if p_val_ks_lipo > alpha else '(ненормальний)'}")
print(f"   - Гемоглобін: p-value = {p_val_ks_hemo:.4f} {'(нормальний)' if p_val_ks_hemo > alpha else '(ненормальний)'}")

# Тест Шапіро-Вілка для додаткової перевірки
stat_sw_lipo, p_val_sw_lipo = shapiro(df['Ліпопротеїни'])
stat_sw_hemo, p_val_sw_hemo = shapiro(df['Гемоглобін'])

print("   Тест Шапіро-Вілка:")
print(f"   - Ліпопротеїни: p-value = {p_val_sw_lipo:.4f} {'(нормальний)' if p_val_sw_lipo > alpha else '(ненормальний)'}")
print(f"   - Гемоглобін: p-value = {p_val_sw_hemo:.4f} {'(нормальний)' if p_val_sw_hemo > alpha else '(ненормальний)'}")

# 3. Візуалізація даних
plt.figure(figsize=(15, 5))

# Діаграма розсіювання
plt.subplot(1, 3, 1)
plt.scatter(df['Ліпопротеїни'], df['Гемоглобін'], alpha=0.6, color='blue')
plt.xlabel('Ліпопротеїни')
plt.ylabel('Гемоглобін')
plt.title('Діаграма розсіювання\nЛіпопротеїни vs Гемоглобін')
plt.grid(True, alpha=0.3)

# Гістограма ліпопротеїнів
plt.subplot(1, 3, 2)
plt.hist(df['Ліпопротеїни'], bins=20, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Ліпопротеїни')
plt.ylabel('Частота')
plt.title('Розподіл ліпопротеїнів')
plt.grid(True, alpha=0.3)

# Гістограма гемоглобіну
plt.subplot(1, 3, 3)
plt.hist(df['Гемоглобін'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Гемоглобін')
plt.ylabel('Частота')
plt.title('Розподіл гемоглобіну')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Кореляційний аналіз
print("\n3. КОРЕЛЯЦІЙНИЙ АНАЛІЗ:")

# Коефіцієнт Пірсона (для порівняння)
correlation_pearson, p_value_pearson = stats.pearsonr(df['Ліпопротеїни'], df['Гемоглобін'])

# Коефіцієнт Спірмена (основний метод)
correlation_spearman, p_value_spearman = stats.spearmanr(df['Ліпопротеїни'], df['Гемоглобін'])

print("   Коефіцієнт кореляції Пірсона:")
print(f"   - r = {correlation_pearson:.3f}, p-value = {p_value_pearson:.4f}")

print("   Коефіцієнт кореляції Спірмена:")
print(f"   - r = {correlation_spearman:.3f}, p-value = {p_value_spearman:.4f}")

# 5. Детальна інтерпретація результатів
print("\n4. ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ:")

# Вибір основного методу на основі нормальності
if p_val_ks_lipo > alpha and p_val_ks_hemo > alpha:
    main_correlation = correlation_pearson
    main_method = "Пірсона"
    print("   Основний метод: кореляція Пірсона (обидві змінні нормальні)")
else:
    main_correlation = correlation_spearman
    main_method = "Спірмена"
    print("   Основний метод: кореляція Спірмена (наявні ненормальні розподіли)")

# Визначення статистичної значущості
if p_value_spearman < alpha:
    significance = "статистично значущий"
else:
    significance = "не є статистично значущим"

# Визначення сили зв'язку
abs_r = abs(main_correlation)
if 0.75 <= abs_r <= 1.00:
    strength = "дуже високий"
    strength_level = "дуже сильний"
elif 0.50 <= abs_r < 0.75:
    strength = "високий" 
    strength_level = "сильний"
elif 0.25 <= abs_r < 0.50:
    strength = "середній"
    strength_level = "помірний"
elif 0.10 <= abs_r < 0.25:
    strength = "слабкий"
    strength_level = "слабкий"
else:
    strength = "дуже слабкий"
    strength_level = "практично відсутній"

# Визначення напрямку зв'язку
if main_correlation < 0:
    direction = "негативний (обернений)"
    interpretation = "збільшення ліпопротеїнів пов'язане зі зменшенням гемоглобіну"
else:
    direction = "позитивний (прямий)" 
    interpretation = "збільшення ліпопротеїнів пов'язане зі збільшенням гемоглобіну"

# Коефіцієнт детермінації
r_squared = main_correlation ** 2

print(f"\n   ОСНОВНІ РЕЗУЛЬТАТИ:")
print(f"   - Метод: кореляція {main_method}")
print(f"   - Коефіцієнт кореляції (r): {main_correlation:.3f}")
print(f"   - Напрямок зв'язку: {direction}")
print(f"   - Сила зв'язку: {strength} ({strength_level})")
print(f"   - Коефіцієнт детермінації (R²): {r_squared:.3f}")
print(f"   - Статистична значущість: {significance} (p < {alpha})")

print(f"\n   ІНТЕРПРЕТАЦІЯ:")
print(f"   - {interpretation}")
print(f"   - {r_squared:.1%} дисперсії гемоглобіну пояснюється змінами ліпопротеїнів")