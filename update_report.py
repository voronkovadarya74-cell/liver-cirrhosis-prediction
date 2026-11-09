import pandas as pd
import os

# === ПУТИ ===
CSV_PATH = os.path.join('reports', 'model_comparison.csv')
MD_PATH = os.path.join('docs', 'hypotheses_and_insights.md')

# === МАРКЕРЫ (должны совпадать с теми, что в Markdown-файле) ===
START_MARKER = '<!-- TABLE_START -->'
END_MARKER = '<!-- TABLE_END -->'

# === ЧТЕНИЕ CSV ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f'Файл {CSV_PATH} не найден. Сначала запусти main.py')

df = pd.read_csv(CSV_PATH)

# === ФОРМАТИРОВАНИЕ ТАБЛИЦЫ ===
# Находим лучшую модель по LogLoss (чем меньше, тем лучше)
best_idx = df['Val LogLoss'].idxmin()
best_model = df.loc[best_idx, 'Model']

# Формируем строки таблицы
header = '| Модель | Val Accuracy | Val F1-macro | Val LogLoss |'
separator = '| :--- | :---: | :---: | :---: |'
rows = []

for i, row in df.iterrows():
    model_name = row['Model']
    acc = f"{row['Val Accuracy']:.4f}"
    f1 = f"{row['Val F1-macro']:.4f}"
    ll = f"{row['Val LogLoss']:.4f}"
    
    # Выделяем лучшую модель жирным
    if model_name == best_model:
        line = f'| **{model_name}** | **{acc}** | **{f1}** | **{ll}** |'
    else:
        line = f'| {model_name} | {acc} | {f1} | {ll} |'
    rows.append(line)

# Собираем полную таблицу
new_table = '\n'.join([header, separator] + rows)

# === ОБНОВЛЕНИЕ MARKDOWN-ФАЙЛА ===
if not os.path.exists(MD_PATH):
    raise FileNotFoundError(f'Файл {MD_PATH} не найден.')

with open(MD_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Проверяем, что маркеры есть
if START_MARKER not in content or END_MARKER not in content:
    raise ValueError(
        f'Маркеры {START_MARKER} и {END_MARKER} не найдены в {MD_PATH}.\n'
        'Добавь их в секцию "Сравнение метрик моделей".'
    )

# Заменяем содержимое между маркерами
start_pos = content.find(START_MARKER) + len(START_MARKER)
end_pos = content.find(END_MARKER)

new_content = (
    content[:start_pos] +
    '\n' + new_table + '\n' +
    content[end_pos:]
)

# Сохраняем
with open(MD_PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('✅ Отчёт успешно обновлён!')
print(f'   Лучшая модель по LogLoss: {best_model}')
print(f'   Файл: {MD_PATH}')