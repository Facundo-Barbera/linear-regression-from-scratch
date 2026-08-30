import csv
import os

from linear_regression import LinearRegression

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'salary.csv')


def load_data(path):
    x = []
    y = []
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y


def split_data(x, y, percentage: float = 0.8):
    split = int(len(x) * percentage)
    return x[:split], y[:split], x[split:], y[split:]


if __name__ == '__main__':
    x, y = load_data(DATA_PATH)
    x_train, y_train, x_test, y_test = split_data(x, y)
    print(f'{len(x)} observaciones, {len(x_train)} de entrenamiento y {len(x_test)} de prueba')

    model = LinearRegression([[x_train], y_train], epochs=100000)
    model.fit()
    print(model)

    y_pred = model.predict([x_test])
    print(f'MSE: {model._mse(y_test, y_pred):.2f}')

    print('\nPredicciones:')
    for i in range(len(x_test)):
        print(f'{x_test[i]} años -> real: {y_test[i]:.0f}, predicho: {y_pred[i]:.0f}')

    years = 12.0
    print(f'\nPrediccion para {years} años de experiencia: {model.predict([[years]])[0]:.0f}')
