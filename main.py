import random

from src.data import (DATA_PATH, FEATURES, load_data, original_scale_params,
                      split_data, standardize)
from src.linear_regression import LinearRegression
from src.metrics import evaluate

SEED = 42
EPOCHS = 10000


def main():
    random.seed(SEED)

    x, y = load_data(DATA_PATH)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)
    print(f'{len(y)} observaciones con {len(x)} variables.')
    print(f'{len(y_train)} para entrenar, {len(y_val)} para validar y {len(y_test)} para probar.')

    x_test_original = x_test
    x_train, estadisticos = standardize(x_train)
    x_val, _ = standardize(x_val, estadisticos)
    x_test, _ = standardize(x_test, estadisticos)

    modelo = LinearRegression([x_train, y_train], epochs=EPOCHS)
    modelo.fit()
    print(f'\nTasa de aprendizaje automatica: {modelo.learning_rate}')

    betas, intercepto = original_scale_params(modelo.betas, modelo.bias, estadisticos)
    print('\nCoeficientes en unidades originales:')
    for nombre, beta in zip(FEATURES, betas):
        print(f'{nombre}: {beta:+.4f}')
    print(f'Intercepto: {intercepto:.4f}')

    print('\nMetricas por conjunto:')
    conjuntos = {'Entrenamiento': (x_train, y_train),
                 'Validacion': (x_val, y_val),
                 'Prueba': (x_test, y_test)}
    for nombre, (entrada, objetivo) in conjuntos.items():
        medidas = evaluate(objetivo, modelo.predict(entrada))
        print(f'{nombre}: MSE {medidas["MSE"]:.2f}, RMSE {medidas["RMSE"]:.3f}, '
              f'MAE {medidas["MAE"]:.3f}, R2 {medidas["R2"]:.4f}')

    print('\nDiez predicciones del conjunto de prueba:')
    y_predicho = modelo.predict(x_test)
    for i in range(10):
        condiciones = ', '.join(f'{nombre} {x_test_original[j][i]:.2f}'
                                for j, nombre in enumerate(FEATURES))
        print(f'Con {condiciones}: real {y_test[i]:.2f} MW, predicho {y_predicho[i]:.2f} MW')


if __name__ == '__main__':
    main()
