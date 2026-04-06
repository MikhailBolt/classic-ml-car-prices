import matplotlib.pyplot as plt
import seaborn as sns

# ... (после обучения модели и расчета preds) ...

# 6. Визуализация результатов (График предсказаний)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=preds)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

# 7. Интерактивное предсказание
print("\n--- Try it yourself! ---")
year = int(input("Enter year (e.g. 2020): "))
mileage = int(input("Enter mileage (e.g. 50000): "))
engine = float(input("Enter engine size (e.g. 2.5): "))

new_car = pd.DataFrame([[year, mileage, engine]], columns=['year', 'mileage', 'engine_size'])
result = model.predict(new_car)
print(f"Predicted price for your car: ${result[0]:.2f}")
