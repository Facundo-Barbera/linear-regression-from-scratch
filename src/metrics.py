import math


def mse(y_true, y_pred):
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_true))) / len(y_true)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return sum(abs(y_pred[i] - y_true[i]) for i in range(len(y_true))) / len(y_true)


def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_true)))
    ss_tot = sum((value - mean_y) ** 2 for value in y_true)
    return 1 - ss_res / ss_tot


def evaluate(y_true, y_pred):
    """Las cuatro metricas del reporte, en un diccionario."""
    error = mse(y_true, y_pred)
    return {
        'MSE': error,
        'RMSE': math.sqrt(error),
        'MAE': mae(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }
