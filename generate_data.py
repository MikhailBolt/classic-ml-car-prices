import pandas as pd
import numpy as np
import os

# Создаем папку data, если ее нет
os.makedirs('data', exist_ok=True)

# Генерируем 1000 машин
n_samples = 1000
data = {
    'year': np.random.randint(2010, 2024, n_samples),
    'mileage': np.random.randint(0, 200000, n_samples),
    'engine_size': np.random.uniform(1.2, 5.0, n_samples),
}

# Цена = (Базовая цена) + (год * 1000) - (пробег * 0.1) + (двигатель * 5000) + шум
df = pd.DataFrame(data)
df['price'] = 20000 + (df['year'] - 2010) * 1500 - (df['mileage'] * 0.15) + (df['engine_size'] * 4000) + np.random.normal(0, 2000, n_samples)

df.to_csv('data/car_prices.csv', index=False)
print("--- Dataset created at data/car_prices.csv ---")
