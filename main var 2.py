import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform
import os

output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

def get_bins(data):
    if len(data) <= 1:
        return 5
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.sqrt(len(data)))
    bin_width = 2 * iqr / (len(data) ** (1/3))
    data_range = np.max(data) - np.min(data)
    if bin_width == 0:
        return int(np.sqrt(len(data)))
    bins = int(data_range / bin_width)
    return max(5, min(bins, 50))

sample_sizes = [10, 100, 1000]
np.random.seed(42)

# ====== 1. НОРМАЛЬНОЕ ======
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Нормальное распределение N(x; 0, 1)', fontsize=16, y=1.05)
for idx, n in enumerate(sample_sizes):
    ax = axes[idx]
    data = norm.rvs(loc=0, scale=1, size=n)
    bins = get_bins(data)
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    x_range = np.linspace(-4, 4, 500)
    ax.plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2)
    ax.set_title(f'n = {n}')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'normal_histograms.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====== 2. КОШИ (ИСПРАВЛЕННАЯ ВЕРСИЯ) ======
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Распределение Коши C(x; 0, 1)', fontsize=16, y=1.05)
for idx, n in enumerate(sample_sizes):
    ax = axes[idx]
    data = cauchy.rvs(loc=0, scale=1, size=n)
    # Фильтруем выбросы для визуализации
    data_filtered = data[(data >= -10) & (data <= 10)]
    bins = get_bins(data_filtered)
    ax.hist(data_filtered, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    x_range = np.linspace(-8, 8, 1000)
    ax.plot(x_range, cauchy.pdf(x_range, 0, 1), 'r-', linewidth=2)
    ax.set_xlim(-8, 8)
    ax.set_title(f'n = {n}')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, linestyle='--', alpha=0.6)
    outliers = len(data) - len(data_filtered)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cauchy_histograms.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====== 3. ЛАПЛАСА ======
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Распределение Лапласа L(x; 0, 1/√2)', fontsize=16, y=1.05)
scale = 1/np.sqrt(2)
for idx, n in enumerate(sample_sizes):
    ax = axes[idx]
    data = laplace.rvs(loc=0, scale=scale, size=n)
    bins = get_bins(data)
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    x_range = np.linspace(-4, 4, 500)
    ax.plot(x_range, laplace.pdf(x_range, 0, scale), 'r-', linewidth=2)
    ax.set_title(f'n = {n}')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'laplace_histograms.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====== 4. ПУАССОНА ======
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Распределение Пуассона P(k; 10)', fontsize=16, y=1.05)
for idx, n in enumerate(sample_sizes):
    ax = axes[idx]
    data = poisson.rvs(10, size=n)
    bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    k_range = np.arange(0, 25)
    ax.bar(k_range, poisson.pmf(k_range, 10), alpha=0.8, color='red', width=0.8)
    ax.set_title(f'n = {n}')
    ax.set_xlabel('k')
    ax.set_ylabel('Вероятность')
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'poisson_histograms.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====== 5. РАВНОМЕРНОЕ ======
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Равномерное распределение U(x; -√3, √3)', fontsize=16, y=1.05)
loc = -np.sqrt(3)
scale = 2 * np.sqrt(3)
for idx, n in enumerate(sample_sizes):
    ax = axes[idx]
    data = uniform.rvs(loc=loc, scale=scale, size=n)
    bins = get_bins(data)
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    x_range = np.linspace(-2.5, 2.5, 500)
    ax.plot(x_range, uniform.pdf(x_range, loc, scale), 'r-', linewidth=2)
    ax.set_title(f'n = {n}')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'uniform_histograms.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Все 5 графиков успешно сохранены в папку 'plots/'")