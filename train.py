import pandas as pd
import jobname # для сохранения модели: pip install joblib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Загрузка
df = pd.read_csv('data/car_prices.csv')

# 2. Признаки и Цель
X = df.drop('price', axis=1)
y = df['price']

# 3. Сплит данных 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Обучение
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Оценка
preds = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, preds):.2f}")
print(f"Mean Error: ${mean_absolute_error(y_test, preds):.2f}")

# 6. Сохранение
joblib.dump(model, 'src/car_model.pkl')
print("--- Model saved to src/car_model.pkl ---")
