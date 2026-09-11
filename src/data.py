import csv
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'datasets', 'powerplant.csv')
FIGURES = os.path.join(ROOT, 'report', 'figures')

FEATURES = ['AT', 'V', 'AP', 'RH']
TARGET = 'PE'


def load_data(path=DATA_PATH, features=None, target: str = TARGET, drop_duplicates: bool = True):
    """Lee el CSV y devuelve (x, y), donde x es una lista por cada feature."""
    features = FEATURES if features is None else features
    columns = {name: [] for name in features}
    y = []
    seen = set()
    with open(path) as file:
        reader = csv.reader(file)
        header = next(reader)
        indices = {name: header.index(name) for name in features}
        target_index = header.index(target)
        for row in reader:
            key = tuple(row)
            if drop_duplicates and key in seen:
                continue
            seen.add(key)
            for name in features:
                columns[name].append(float(row[indices[name]]))
            y.append(float(row[target_index]))
    return [columns[name] for name in features], y


def split_data(x, y, train_pct: float = 0.6, val_pct: float = 0.2):
    """Reparte en entrenamiento, validacion y prueba barajando los indices.

    El archivo conserva el orden de captura original, asi que sin barajar los
    tres conjuntos vendrian de periodos distintos.
    """
    indices = list(range(len(y)))
    random.shuffle(indices)
    train_end = int(len(y) * train_pct)
    val_end = int(len(y) * (train_pct + val_pct))

    def take(selected):
        return [[feature[i] for i in selected] for feature in x], [y[i] for i in selected]

    x_train, y_train = take(indices[:train_end])
    x_val, y_val = take(indices[train_end:val_end])
    x_test, y_test = take(indices[val_end:])
    return x_train, y_train, x_val, y_val, x_test, y_test


def standardize(x, stats=None):
    """Escala cada feature a media 0 y desviacion 1.

    Las variables tienen magnitudes muy distintas (AP ~1013 frente a AT ~20), y
    el descenso de gradiente usa un mismo paso para todas. Sin escalar, el paso
    tiene que ser diminuto para que AP no se dispare, y entonces las demas casi
    no avanzan.

    Pasa `stats` para reutilizar los estadisticos del entrenamiento en
    validacion y prueba; calcularlos con todos los datos filtraria informacion
    de los conjuntos de evaluacion.
    """
    if stats is None:
        stats = []
        for feature in x:
            n = len(feature)
            mean = sum(feature) / n
            std = (sum((value - mean) ** 2 for value in feature) / n) ** 0.5
            stats.append((mean, std))
    scaled = [[(value - mean) / std for value in feature]
              for feature, (mean, std) in zip(x, stats)]
    return scaled, stats


def original_scale_params(betas, bias, stats):
    """Convierte los parametros del espacio escalado a las unidades originales."""
    betas_original = [beta / std for beta, (mean, std) in zip(betas, stats)]
    bias_original = bias - sum(beta * mean / std for beta, (mean, std) in zip(betas, stats))
    return betas_original, bias_original
