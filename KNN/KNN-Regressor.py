# K-Nearest Neighbors Regressor

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class KNNRegressor:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indexes = np.argsort(distances)[:self.k]
        k_nearest_values = self.y_train[k_indexes]
        return np.mean(k_nearest_values)
    

    def plot_regression_results(self, y_true, y_pred, title):
        plt.figure(figsize=(8, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Previsões do Modelo')
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Previsão Ideal")

        plt.xlabel('Valor Real')
        plt.ylabel('Valor Previsto')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.savefig(f"KNN/imgs/knn-regressor-{title.replace(" ", "-").replace(":", "-")}.jpg",
                dpi=300,
                bbox_inches='tight')
        plt.show()
        plt.close()



## Tests
# Dataset 1
print("\n--- Testando com Dataset Diabetes ---")
data = load_diabetes()
X, y = data.data, data.target

# 30% de conjunto de testes
# 70% de conjunto de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_regressor = KNNRegressor(k=5)
knn_regressor.fit(X_train, y_train)

predictions = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
knn_regressor.plot_regression_results(y_test, predictions, title="Diabetes: Real vs Previsto")


# Dataset 2
print("--- Testando com Dataset California Housing")
housing = fetch_california_housing()

X_h, y_h = housing.data, housing.target

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

knn_housing = KNNRegressor(k=5)
knn_housing.fit(X_train_h, y_train_h)
preds_housing = knn_housing.predict(X_test_h)
print(f"MSE Housing: {mean_squared_error(y_test_h, preds_housing):.2f}")
knn_housing.plot_regression_results(y_test_h, preds_housing, title="California Housing: Real vs Previsto")