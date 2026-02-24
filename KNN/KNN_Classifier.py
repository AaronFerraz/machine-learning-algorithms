# K-Nearest Neighbors Classifier

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from collections import Counter

class KNNClassifier:
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
        k_nearest_labels = self.y_train[k_indexes]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

    def plot_iris(self, X_train, y_train, X_test, y_test, y_pred, target_names, accuracy):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            mask = (y_test == i)
            if mask.any():
                ax1.scatter(X_test[mask, 0], X_test[mask, 1],
                        c=color, label=target_names[i],
                        edgecolor='k', s=50)
        
        ax1.set_xlabel('Comprimento da Sépala (cm)')
        ax1.set_ylabel('Largura da Sépala (cm)')
        ax1.set_title('Visualização Simples do Dataset Iris')
        ax1.legend()
        ax1.grid(True, alpha=0.6)
        
        for i, color in enumerate(colors):
            mask = (y_pred == i)
            if mask.any():
                ax2.scatter(X_test[mask, 0], X_test[mask, 1],
                        c=color, label=target_names[i],
                        edgecolor='k', s=50, alpha=0.7)
        
        ax2.set_xlabel('Comprimento da Sépala (cm)')
        ax2.set_ylabel('Largura de Sépala (cm)')
        ax2.set_title(f'Dataset Predito pelo KNN\nAcurácia: {accuracy:.2%}')
        ax2.legend()
        ax2.grid(True, alpha=0.6)

        plt.suptitle("Comparação: Original vs Predito - Iris Dataset", fontsize=14)
        plt.tight_layout()
        plt.savefig("KNN/imgs/knn-classicador.jpg",
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()



# Tests
print("--- Dataset de Flores ---")
data = load_iris()
X, y, target_names = data.data, data.target, data.target_names

# 50% de conjunto de testes
# 50% de conjunto de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Acurácia do KNN foi: {accuracy * 100:.2f}%")

knn.plot_iris(X_train, y_train, X_test, y_test, predictions, target_names, accuracy)
