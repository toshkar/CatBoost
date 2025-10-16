''' Итоговая цель
Код создает систему для прогнозирования направления движения акций Сбербанка на следующий торговый день на основе исторических данных о ценах и объемах торгов.

1. Сбор и обработка данных с Московской биржи
Что делает:
Получает данные по акциям Сбербанка (SBER) с Московской биржи за период с 1 января 2024 по 22 апреля 2025
Собирает торговую статистику: цены открытия, закрытия, максимумы, минимумы, объемы торгов
Вычисляет дельту - разницу между объемом покупок и продаж в процентах

Создает целевую переменную res_next, которая показывает:
1 если цена следующего дня выросла (close > open)
-1 если цена упала
Особенности обработки:
Данные агрегируются по дням
Создаются признаки для модели машинного обучения:
close, high, low - относительные изменения цен относительно открытия
дельта - индикатор соотношения покупок/продаж

2. Обучение модели CatBoost
Что делает:
Обучает модель классификации CatBoost для прогнозирования направления движения цены
Использует временные ряды с сохранением порядка данных
Оценивает точность предсказаний

Ключевые особенности модели:
boosting_type='Ordered' и has_time=True - учитывает временную природу данных
shuffle=False - сохраняет временной порядок при разделении на train/test
Ранняя остановка для предотвращения переобучения
Анализ важности признаков '''

import pandas as pd
from statsmodels.formula.api import ols
from moexalgo import Ticker, session

# Авторизация на Московской бирже
# session.TOKEN

# Получение данных по Сбербанку
ticker = 'SBER'
ticker_2 = Ticker(ticker)
start_date = '2024-01-01'
end_date = '2025-04-22'
data = ticker_2.tradestats(start=start_date, end=end_date)

# Обработка данных
columns_to_include = ['tradedate', 'tradetime', 'pr_open', 'pr_high', 'pr_low', 'pr_close', 'vol', 'pr_vwap', 'vol_b', 'vol_s']
data_2 = data[columns_to_include]
data_2['tradedate'] = pd.to_datetime(data_2['tradedate'])

# Агрегация по дням
data_3 = data_2.groupby('tradedate').agg(
    pr_open=('pr_open', 'first'),
    pr_high=('pr_high', 'max'),
    pr_low=('pr_low', 'min'),
    pr_close=('pr_close', 'last'),
    pr_vwap=('pr_vwap', 'mean'),  # Исправлено - добавлена функция агрегации
    vol=('vol', 'sum'),
    vol_b=('vol_b', 'sum'),
    vol_s=('vol_s', 'sum')
)

# Расчет дельты и подготовка признаков
data_3['дельта'] = (data_3['vol_b'] - data_3['vol_s']) / data_3['vol'] * 100
data_3 = data_3[['pr_open', 'pr_high', 'pr_low', 'pr_close', 'дельта']]

# Создание целевой переменной (движение цены на следующий день)
data_3['close_+1'] = data_3['pr_close'].shift(-1)
data_3['open_+1'] = data_3['pr_open'].shift(-1)
data_3['res_next'] = data_3.apply(lambda row: 1 if row['close_+1'] - row['open_+1'] > 0 else -1, axis=1)
data_3 = data_3.iloc[:-1]

# Создание признаков для модели
data_3['close'] = data_3['pr_close'] - data_3['pr_open']
data_3['high'] = data_3['pr_high'] - data_3['pr_open']
data_3['low'] = data_3['pr_low'] - data_3['pr_open']

# Обучение модели CatBoost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Загрузка данных (здесь предполагается, что данные сохранены в файл)
df = pd.read_excel('data_candles.xlsx')

# Разделение на признаки и целевую переменную
X = df.drop('res_next', axis=1)
y = df['res_next']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Создание и обучение модели
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

model = CatBoostClassifier(
    iterations=1000,
    boosting_type='Ordered',
    has_time=True,
    loss_function='Logloss',
    eval_metric='Accuracy',
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# Оценка модели
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Анализ важности признаков
feature_importances = model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
