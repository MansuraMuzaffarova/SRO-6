# SRO-6
# Импорт необходимых библиотек
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка набора данных (в данном случае, набор данных Iris)
data = load_iris()
X = data.data
y = data.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание экземпляра решающего дерева
clf = tree.DecisionTreeClassifier()

# Обучение решающего дерева на тренировочных данных
clf.fit(X_train, y_train)

# Предсказание на тестовых данных
predictions = clf.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, predictions)
print("Точность модели: {:.2f}%".format(accuracy * 100))
