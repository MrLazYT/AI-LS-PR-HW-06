import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Завантаження даних
df = pd.read_csv('internship_candidates_cefr_final.csv')

# Перетворення рівня англійської у числові значення
english_map = {
    'Elementary': 1,
    'Pre-Intermediate': 2,
    'Intermediate': 3,
    'Upper-Intermediate': 4,
    'Advanced': 5,
    'Proficient': 6
}
df['EnglishLevelNum'] = df['EnglishLevel'].map(english_map)

# Вибір ознак і цільової змінної
X = df[["Experience", "Grade", "EnglishLevelNum", "Age", "EntryTestScore"]]
y = df["Accepted"]

# Розбиття на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення пайплайну з масштабуванням і логістичною регресією
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Навчання моделі
model.fit(X_train, y_train)

# Прогноз на тестовій вибірці
y_pred = model.predict(X_test)

# Побудова графіка ймовірності прийняття залежно від EntryTestScore
score_range = np.linspace(300, 1000, 100)

# Тестовий DataFrame для графіку
plot_df = pd.DataFrame({
    'Experience': [3]*len(score_range),
    'Grade': [9.0]*len(score_range),
    'EnglishLevelNum': [3]*len(score_range),
    'Age': [23]*len(score_range),
    'EntryTestScore': score_range
})

# Прогноз імовірностей
probs = model.predict_proba(plot_df)[:, 1]

# Побудова графіка
plt.plot(score_range, probs)
plt.xlabel('Entry Test Score')
plt.ylabel('Probability of Acceptance')
plt.title('Acceptance Probability vs Entry Test Score')
plt.grid(True)
plt.show()