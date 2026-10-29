"""Neural Network Training Lab — учебный симулятор обучения нейронной сети.

Шестой и завершающий проект серии образовательных симуляторов:
    1. One Artificial Neuron      2. Perceptron        3. Mini Neural Network
    4. Gradient Descent           5. Backpropagation   6. Neural Network Training Lab

Здесь объясняется не один шаг обучения, а повторяющийся процесс:
    датасет -> прямой проход -> предсказание -> потеря -> обратный проход
    -> градиенты -> обновление весов -> следующий пример -> следующая эпоха

Сеть фиксирована: 2 -> 2 -> 1, сигмоида в обоих слоях, девять обучаемых
параметров, стохастическое обновление (один шаг на каждый пример).

Только стандартная библиотека: Python 3, tkinter, math, random.
Внешние пакеты (NumPy, matplotlib и другие) не используются.

Автор: SteveWorks001

Куда смотреть
-------------
Раздел 1   Ограничения ввода, палитра, обозначения            (строки ~40)
Раздел 2   Форматирование чисел
Раздел 3   Математика сети: sigmoid, forward, loss, backward
Раздел 4   Датасеты и начальные параметры
Раздел 5   Обучение: эпоха, история потерь, класс Trainer
Раздел 6   Пошаговый режим: этапы и текстовые блоки
Раздел 7   Локализация RU/EN
Раздел 8   Виджеты: поле параметра, схема сети, график потерь, прокрутка
Раздел 9   Главное окно приложения
Раздел 10  Точка входа
"""

import math
import random
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ---------------------------------------------------------------
# 1. Ограничения ввода, палитра, обозначения
# ---------------------------------------------------------------

# Диапазоны полей ввода. Выход за границу не роняет приложение: значение
# приводится к ближайшему допустимому, а поле подсвечивается.
LIMITS = {
    "epochs": (1, 10000),
    "eta": (0.0, 1.5),
    "w": (-5.0, 5.0),   # веса и смещения, редактируемые пользователем
}

# Скорость автоматического обучения: (эпох за один шаг, задержка мс).
SPEEDS = {
    "slow": (1, 120),
    "normal": (3, 30),
    "fast": (20, 0),
}

COL_TEXT = "#242424"        # основной текст
COL_BG = "#fbfbf7"          # фон окна
COL_PAPER = "#fdfdf9"       # «бумажные» панели: canvas, tk.Text
COL_FWD = "#3f6f96"         # прямой проход (синий)
COL_BWD = "#b3261e"         # обратный проход (красный)
COL_NODE = "#eef3f7"        # узел без подсветки
COL_NODE_LINE = "#8d949b"
COL_ACTIVE = "#b07d10"      # янтарная подсветка текущего этапа
COL_ACTIVE_FILL = "#fdf1d8"
COL_DIM = "#9a9a93"
COL_FRAME = "#c8c8c0"       # рамки
COL_POS = "#2f7d4f"         # рост/успех (зелёный)
COL_NEG = "#b3261e"         # падение/ошибка (красный)
COL_BAD_BG = "#fde8e6"      # фон некорректного поля ввода
COL_BAD_FG = "#7a1a13"
COL_GRID = "#dcdcd4"        # сетка графика
COL_CURVE = "#3f6f96"       # кривая средних потерь
COL_CURVE_PT = "#b07d10"

# Девять обучаемых параметров в едином порядке (порядок важен для таблиц).
HIDDEN1_KEYS = ("w1", "w2", "b1")
HIDDEN2_KEYS = ("w3", "w4", "b2")
OUTPUT_KEYS = ("w5", "w6", "bo")
PARAM_KEYS = HIDDEN1_KEYS + HIDDEN2_KEYS + OUTPUT_KEYS
WEIGHT_KEYS = tuple(k for k in PARAM_KEYS if not k.startswith("b"))
BIAS_KEYS = ("b1", "b2", "bo")

# Обозначения на схеме: (откуда, куда) -> имя параметра.
EDGE_PARAM = {
    ("x1", "h1"): "w1", ("x2", "h1"): "w2",
    ("x1", "h2"): "w3", ("x2", "h2"): "w4",
    ("h1", "o"): "w5", ("h2", "o"): "w6",
}
# Ключевые производные, которые показывает пошаговый режим.
PARTIAL = "∂"   # символ производной


def d(name):
    """Имя производной: ∂L/∂w1."""
    return PARTIAL + "L/" + PARTIAL + name


GRAD_SYM = {k: d(k) for k in PARAM_KEYS}

MINUS = "−"   # типографский минус только для вывода; ввод остаётся ASCII


# ---------------------------------------------------------------
# 2. Форматирование чисел
# ---------------------------------------------------------------

def f4(value):
    """Число с четырьмя знаками после запятой, без «-0.0000»."""
    if value is None:
        return "—"
    v = float(value)
    if abs(v) < 5e-5:
        v = 0.0
    return ("%.4f" % v).replace("-", MINUS)


def f2(value):
    if value is None:
        return "—"
    v = float(value)
    if abs(v) < 5e-3:
        v = 0.0
    return ("%.2f" % v).replace("-", MINUS)


def fmt(value, digits=4):
    if value is None:
        return "—"
    v = float(value)
    text = ("%%.%df" % digits) % v
    if float(text) == 0.0:
        text = "0." + "0" * digits
    return text.replace("-", MINUS)


def signed(value, digits=4):
    """Явный знак: удобно читать градиенты и приращения."""
    v = float(value)
    if abs(v) < 5e-5:
        return fmt(0.0, digits)
    return ("+" if v > 0 else MINUS) + fmt(abs(v), digits)


def mul(a, b):
    """Строка вида «0.5·1» для подстановки чисел в формулу."""
    return "%s·%s" % (fmt(a), fmt(b))


def parse_number(raw):
    """Разбирает «0.1», «0,1», «-2», « 3 »; возвращает float или None."""
    if raw is None:
        return None
    text = str(raw).strip().replace(",", ".").replace(MINUS, "-").replace(" ", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def clamp(value, low, high):
    return min(high, max(low, value))


# ---------------------------------------------------------------
# 3. Математика сети 2 -> 2 -> 1
# ---------------------------------------------------------------

def sigmoid(z):
    """Численно устойчивая сигмоида: exp() вызывается только от неположительного
    аргумента, поэтому переполнение невозможно даже при |z| ~ 1000."""
    z = float(z)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def sigmoid_derivative(a):
    """σ'(z) = σ(z)·(1 − σ(z)); a — уже вычисленное σ(z)."""
    return a * (1.0 - a)


def forward(x1, x2, p):
    """Прямой проход. Возвращает ВСЕ промежуточные величины, а не только ответ.

    z_h1 = x1·w1 + x2·w2 + b1      h1 = σ(z_h1)
    z_h2 = x1·w3 + x2·w4 + b2      h2 = σ(z_h2)
    z_o  = h1·w5 + h2·w6 + bo      ŷ  = σ(z_o)
    """
    z_h1 = x1 * p["w1"] + x2 * p["w2"] + p["b1"]
    h1 = sigmoid(z_h1)
    z_h2 = x1 * p["w3"] + x2 * p["w4"] + p["b2"]
    h2 = sigmoid(z_h2)
    z_o = h1 * p["w5"] + h2 * p["w6"] + p["bo"]
    y_hat = sigmoid(z_o)
    return {
        "x1": x1, "x2": x2,
        "z_h1": z_h1, "h1": h1,
        "z_h2": z_h2, "h2": h2,
        "z_o": z_o, "y_hat": y_hat,
    }


def sample_loss(y_hat, y):
    """Потеря одного примера: L = 1/2 (ŷ − y)²."""
    d = y_hat - y
    return 0.5 * d * d


def dataset_loss(rows, p):
    """Средняя потеря по датасету при НЕИЗМЕННЫХ весах (оценочный прогон)."""
    total = 0.0
    for x1, x2, y in rows:
        total += sample_loss(forward(x1, x2, p)["y_hat"], y)
    return total / len(rows) if rows else 0.0


def backward(x1, x2, y, p, act):
    """Обратный проход: цепочка производных и девять градиентов.

    ∂L/∂ŷ = ŷ − y
    ∂ŷ/∂z_o = σ'(z_o) = ŷ(1−ŷ)
    ∂L/∂z_o = ∂L/∂ŷ · ∂ŷ/∂z_o                      (дельта выхода)
    ∂L/∂w5 = ∂L/∂z_o · h1,  ∂L/∂w6 = ∂L/∂z_o · h2,  ∂L/∂bo = ∂L/∂z_o
    ∂L/∂h1 = ∂L/∂z_o · w5,  ∂L/∂h2 = ∂L/∂z_o · w6
    ∂L/∂z_h1 = ∂L/∂h1 · σ'(z_h1),  ∂L/∂z_h2 = ∂L/∂h2 · σ'(z_h2)
    ∂L/∂w1 = ∂L/∂z_h1 · x1, ∂L/∂w2 = ∂L/∂z_h1 · x2, ∂L/∂b1 = ∂L/∂z_h1
    ∂L/∂w3 = ∂L/∂z_h2 · x1, ∂L/∂w4 = ∂L/∂z_h2 · x2, ∂L/∂b2 = ∂L/∂z_h2
    """
    d_loss_d_y = act["y_hat"] - y
    d_y_d_zo = sigmoid_derivative(act["y_hat"])
    delta_o = d_loss_d_y * d_y_d_zo

    d_loss_d_h1 = delta_o * p["w5"]
    d_loss_d_h2 = delta_o * p["w6"]
    d_h1_d_zh1 = sigmoid_derivative(act["h1"])
    d_h2_d_zh2 = sigmoid_derivative(act["h2"])
    delta_h1 = d_loss_d_h1 * d_h1_d_zh1
    delta_h2 = d_loss_d_h2 * d_h2_d_zh2

    grads = {
        "w5": delta_o * act["h1"],
        "w6": delta_o * act["h2"],
        "bo": delta_o,
        "w1": delta_h1 * x1,
        "w2": delta_h1 * x2,
        "b1": delta_h1,
        "w3": delta_h2 * x1,
        "w4": delta_h2 * x2,
        "b2": delta_h2,
    }
    return {
        "d_loss_d_y": d_loss_d_y,
        "d_y_d_zo": d_y_d_zo,
        "delta_o": delta_o,
        "d_loss_d_h1": d_loss_d_h1,
        "d_loss_d_h2": d_loss_d_h2,
        "d_h1_d_zh1": d_h1_d_zh1,
        "d_h2_d_zh2": d_h2_d_zh2,
        "delta_h1": delta_h1,
        "delta_h2": delta_h2,
        "grads": grads,
    }


def updated_params(p, grads, eta):
    """Правило обновления для всех девяти параметров: new = old − η · grad."""
    return {k: p[k] - eta * grads[k] for k in PARAM_KEYS}


# ---------------------------------------------------------------
# 4. Датасеты и начальные параметры
# ---------------------------------------------------------------

DATASET_IDS = ("AND", "OR", "XOR")

DATASETS = {
    "AND": ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)),
    "OR": ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)),
    "XOR": ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
}

# Стандартный пример. Значения маленькие и намеренно несимметричные: при
# одинаковых весах скрытые нейроны схлопываются в один и XOR не разделим.
STANDARD_PARAMS = {
    "w1": 0.60, "w2": 0.45, "b1": -0.10,
    "w3": -0.55, "w4": 0.35, "b2": 0.20,
    "w5": 0.50, "w6": -0.40, "bo": 0.15,
}

RANDOM_WEIGHT_LIMIT = 0.5   # случайные веса берутся из [−0.5, +0.5]


def random_params(rng=None):
    """Простая инициализация: равномерные значения, достаточно маленькие."""
    rng = rng or random
    return {k: round(rng.uniform(-RANDOM_WEIGHT_LIMIT, RANDOM_WEIGHT_LIMIT), 4)
            for k in PARAM_KEYS}


# ---------------------------------------------------------------
# 5. Обучение: эпоха, история, состояние тренировки
# ---------------------------------------------------------------

class Trainer:
    """Вся тренировка в одном месте: параметры, история, счётчики.

    Эпоха = каждый пример датасета обрабатывается по очереди, и после каждого
    примера веса обновляются (стохастическое обновление, один шаг на пример).

    Два разных числа, которые нельзя путать:
      epoch_avg_loss — среднее по датасету потерь, вычисленных В ХОДЕ эпохи
                       (каждая — своими, уже обновлёнными весами);
      eval_loss      — средняя потеря по датасету при текущих, неизменных весах.
    """

    def __init__(self, dataset_id="XOR", params=None, eta=0.1, total_epochs=100):
        self.dataset_id = dataset_id
        self.initial = dict(params if params is not None else STANDARD_PARAMS)
        self.params = dict(self.initial)
        self.eta = eta
        self.total_epochs = total_epochs
        self.epoch = 0                 # завершённых эпох
        self.sample_index = 0          # какой пример обрабатывается/показан
        self.history = []              # [(epoch, epoch_avg_loss), ...]
        self.last_sample_loss = None   # потеря последнего обработанного примера
        self.last_grads = None         # градиенты последнего шага
        self.last_step = None          # полный разбор последнего шага (для вкладок)
        self.status = "ready"          # ready | training | stopped | completed | reset
        self.reset_eval = dataset_loss(DATASETS[dataset_id], self.params)

    # -- состояние ---------------------------------------------------------
    @property
    def rows(self):
        return DATASETS[self.dataset_id]

    @property
    def n_samples(self):
        return len(self.rows)

    @property
    def current_sample(self):
        return self.rows[self.sample_index % self.n_samples]

    @property
    def started(self):
        return self.epoch > 0 or self.sample_index > 0

    def set_dataset(self, dataset_id):
        """Смена датасета: обучение начинается заново, включая возврат весов
        к начальным (история, счётчик эпох и разбор шага очищаются)."""
        self.dataset_id = dataset_id
        self.reset_state()

    def set_params(self, params):
        self.params = dict(params)
        self.reset_eval = dataset_loss(self.rows, self.params)

    def set_initial(self, params):
        """Начальные параметры (то, к чему возвращается Reset)."""
        self.initial = dict(params)
        self.params = dict(params)
        self.history = []
        self.epoch = 0
        self.sample_index = 0
        self.last_sample_loss = None
        self.last_grads = None
        self.last_step = None
        self.status = "reset"
        self.reset_eval = dataset_loss(self.rows, self.params)

    def reset_state(self):
        """Вернуть сеть к начальным параметрам и очистить историю эпох."""
        self.params = dict(self.initial)
        self.epoch = 0
        self.sample_index = 0
        self.history = []
        self.last_sample_loss = None
        self.last_grads = None
        self.last_step = None
        self.status = "reset"
        self.reset_eval = dataset_loss(self.rows, self.params)

    def clear_history(self):
        """Только историю: веса, эпоха и текущий пример не меняются."""
        self.history = []
        self.epoch = 0
        self.reset_eval = dataset_loss(self.rows, self.params)

    # -- один пример -------------------------------------------------------
    def process_sample(self, index):
        """Прямой проход, потеря, обратный проход и обновление всех девяти
        параметров для примера с номером index. Возвращает полный разбор шага."""
        x1, x2, y = self.rows[index]
        act = forward(x1, x2, self.params)
        loss = sample_loss(act["y_hat"], y)
        back = backward(x1, x2, y, self.params, act)
        before = dict(self.params)
        self.params = updated_params(self.params, back["grads"], self.eta)
        step = {
            "index": index, "x1": x1, "x2": x2, "y": y,
            "act": act, "loss": loss, "back": back,
            "before": before, "after": dict(self.params), "eta": self.eta,
        }
        self.last_sample_loss = loss
        self.last_grads = back["grads"]
        self.last_step = step
        return step

    # -- эпоха -------------------------------------------------------------
    def run_epoch(self):
        """Одна полная эпоха: все примеры по порядку, обновление после каждого.

        Потеря каждого примера считается ДО обновления весов по этому примеру,
        поэтому среднее — это «как сеть видела датасет в ходе эпохи».
        """
        losses = []
        for i in range(self.n_samples):
            self.sample_index = i
            losses.append(self.process_sample(i)["loss"])
        self.epoch += 1
        avg = sum(losses) / len(losses)
        self.history.append((self.epoch, avg))
        self.sample_index = 0
        return avg, losses

    def predictions(self):
        """Предсказания текущей сети по всем примерам (оценочный прогон)."""
        out = []
        for x1, x2, y in self.rows:
            y_hat = forward(x1, x2, self.params)["y_hat"]
            out.append({"x1": x1, "x2": x2, "y": y, "y_hat": y_hat,
                        "error": y_hat - y, "loss": sample_loss(y_hat, y)})
        return out

    @property
    def eval_loss(self):
        return dataset_loss(self.rows, self.params)

    @property
    def epoch_avg_loss(self):
        return self.history[-1][1] if self.history else None

    @property
    def best_prediction_count(self):
        """Сколько примеров дадут верный ответ при пороге 0.5."""
        return sum(1 for row in self.predictions()
                   if (row["y_hat"] >= 0.5) == (row["y"] >= 0.5))


# ---------------------------------------------------------------
# 6. Пошаговый режим
# ---------------------------------------------------------------

TOTAL_STEPS = 10


def snapshot_step(trainer):
    """Разбор текущего примера БЕЗ записи в сеть.

    Пошаговый режим объясняет математику на снимке параметров: кнопки
    Next/Previous Step не меняют веса, поэтому ими можно ходить многократно.
    Реальное обучение делают Step Epoch и Train.
    """
    x1, x2, y = trainer.current_sample
    act = forward(x1, x2, trainer.params)
    loss = sample_loss(act["y_hat"], y)
    back = backward(x1, x2, y, trainer.params, act)
    new_params = updated_params(trainer.params, back["grads"], trainer.eta)
    return {
        "index": trainer.sample_index, "x1": x1, "x2": x2, "y": y,
        "act": act, "loss": loss, "back": back,
        "before": dict(trainer.params), "after": new_params, "eta": trainer.eta,
    }


# ---------------------------------------------------------------
# 6. Локализация RU/EN
# ---------------------------------------------------------------

LANG = "ru"

# Формулы не переводятся: математика одинакова в обоих языках. Знак
# производной берётся из d(), чтобы в девяти похожих строках не было опечаток.
FORMULA = {
    "sigmoid": "σ(z) = 1 / (1 + e^(−z))",
    "sigmoid_prime": "σ'(z) = σ(z)·(1 − σ(z))",
    "z_h1": "z_h1 = x1·w1 + x2·w2 + b1",
    "h1": "h1 = σ(z_h1)",
    "z_h2": "z_h2 = x1·w3 + x2·w4 + b2",
    "h2": "h2 = σ(z_h2)",
    "z_o": "z_o = h1·w5 + h2·w6 + bo",
    "y_hat": "ŷ = σ(z_o)",
    "loss": "L = 1/2·(ŷ − y)²",
    "avg_loss": "L_сред = (1/N)·Σ L_i",
    "d_loss_dy": d("ŷ") + " = ŷ − y",
    "d_y_dzo": "∂ŷ/∂z_o = ŷ·(1 − ŷ)",
    "delta_o": d("z_o") + " = " + d("ŷ") + " · ∂ŷ/∂z_o",
    "dw5": d("w5") + " = " + d("z_o") + " · h1",
    "dw6": d("w6") + " = " + d("z_o") + " · h2",
    "dbo": d("bo") + " = " + d("z_o"),
    "dh1": d("h1") + " = " + d("z_o") + " · w5",
    "dh2": d("h2") + " = " + d("z_o") + " · w6",
    "dz_h1": d("z_h1") + " = " + d("h1") + " · σ'(z_h1)",
    "dz_h2": d("z_h2") + " = " + d("h2") + " · σ'(z_h2)",
    "dw1": d("w1") + " = " + d("z_h1") + " · x1",
    "dw2": d("w2") + " = " + d("z_h1") + " · x2",
    "db1": d("b1") + " = " + d("z_h1"),
    "dw3": d("w3") + " = " + d("z_h2") + " · x1",
    "dw4": d("w4") + " = " + d("z_h2") + " · x2",
    "db2": d("b2") + " = " + d("z_h2"),
    "update": "new = old − η · " + PARTIAL + "L/" + PARTIAL + "p",
}


class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _safe_format(template, **kw):
    """Формат без падения на пропущенном ключе: лишний {key} остаётся видимым."""
    try:
        return template.format_map(_Safe(kw))
    except (ValueError, IndexError):
        return template


def T(key, **kw):
    text = TEXTS[LANG].get(key)
    if text is None:
        text = TEXTS["ru"].get(key, key)
    return _safe_format(text, **kw) if kw else text


def set_language(code):
    global LANG
    LANG = code


TEXTS = {
    "ru": {
        "app_title": "Лаборатория обучения нейронной сети",
        "subtitle": "Как сеть меняется от примера к примеру и от эпохи к эпохе",
        "lang_ru": "RU",
        "lang_en": "EN",
        "author_note": "Автор: SteveWorks001",

        "box_dataset": "Датасет",
        "box_sample": "Текущий пример",
        "box_params": "Параметры обучения",
        "box_actions": "Управление обучением",
        "box_weights": "Параметры сети (редактируются)",
        "box_speed": "Скорость обучения",
        "box_overview": "Обзор обучения",

        "ds_AND": "AND (И)",
        "ds_OR": "OR (ИЛИ)",
        "ds_XOR": "XOR (исключающее ИЛИ)",
        "desc_AND": "Истина только когда оба входа равны 1. Линейно разделима.",
        "desc_OR": "Истина, если хотя бы один вход равен 1. Линейно разделима.",
        "desc_XOR": "Истина, когда входы различаются. Линейно НЕ разделима: "
                    "без скрытого слоя сеть не сможет её выучить.",

        "th_sample": "Номер",
        "th_x1": "x1",
        "th_x2": "x2",
        "th_target": "Цель",
        "th_prediction": "Предсказание",
        "th_error": "Ошибка",
        "th_loss": "Потеря",
        "th_param": "Параметр",
        "th_value": "Значение",
        "th_gradient": "Градиент",
        "th_old": "Было",
        "th_new": "Стало",
        "th_change": "Приращение",
        "th_before": "До обучения",
        "th_current": "Сейчас",
        "th_compare": "Сравнение",

        "lbl_sample_of": "Пример {i} / {n}",
        "lbl_x1": "x1 = {v}",
        "lbl_x2": "x2 = {v}",
        "lbl_target": "target = {v}",
        "lbl_epochs": "Эпохи:",
        "lbl_eta": "Скорость обучения η:",
        "lbl_epoch_of": "Эпоха {e} / {total}",
        "lbl_status": "Состояние:",
        "lbl_epoch_avg": "Среднее за эпоху",
        "lbl_eval_loss": "Потеря (тек. веса)",
        "lbl_sample_loss": "Потеря примера",
        "lbl_correct": "Верных ответов: {k} / {n}",
        "lbl_epochs_hint": "1–10000",
        "lbl_eta_hint": "0–1.5, можно через запятую",
        "lbl_dataset_note": "Текущий пример выделен в таблице.",
        "lbl_epoch_def": "Эпоха = все примеры датасета по очереди, обновление весов после каждого примера.",
        "lbl_two_losses": "Среднее за эпоху считается по ходу обучения (свои веса для каждого "
                          "примера), а «Потеря (тек. веса)» — это один оценочный прогон с "
                          "неизменными весами.",

        "btn_step_epoch": "Шаг эпоха",
        "btn_train": "Обучение",
        "btn_stop": "Стоп",
        "btn_reset": "Сброс",
        "btn_reset_history": "Очистить историю",
        "btn_random": "Случайные веса",
        "btn_standard": "Стандартные веса",
        "btn_prev": "Предыдущий шаг",
        "btn_next": "Следующий шаг",
        "btn_reset_steps": "Сброс шага",
        "btn_full": "Весь процесс сразу",

        "speed_slow": "Медленно",
        "speed_normal": "Обычно",
        "speed_fast": "Быстро",

        "grp_h1": "Скрытый нейрон 1",
        "grp_h2": "Скрытый нейрон 2",
        "grp_o": "Нейрон выхода",
        "hint_w1": "x1 → h1",
        "hint_w2": "x2 → h1",
        "hint_b1": "смещение h1",
        "hint_w3": "x1 → h2",
        "hint_w4": "x2 → h2",
        "hint_b2": "смещение h2",
        "hint_w5": "h1 → ŷ",
        "hint_w6": "h2 → ŷ",
        "hint_bo": "смещение ŷ",

        "state_ready": "Готово",
        "state_training": "Идёт обучение",
        "state_stopped": "Остановлено",
        "state_completed": "Завершено",
        "state_reset": "Сброшено",
        "status_ready": "Готово. Выберите датасет и начните обучение.",
        "status_training": "Идёт обучение…",
        "status_stopped": "Остановлено на эпохе {e}.",
        "status_completed": "Обучение завершено: {total} эпох.",
        "status_reset": "Сброс: сеть вернулась к начальным параметрам, история очищена.",
        "status_lang": "Язык изменён. Обучение, веса и история не тронуты.",
        "status_epoch_done": "Эпоха {e}: средняя потеря {loss}.",
        "status_bad": "«{raw}» — не число. Последнее допустимое значение сохранено.",
        "status_range": "{name}: разрешено от {lo} до {hi}. Взято ближайшее допустимое значение.",
        "status_dataset": "Датасет: {name}. Обучение начато заново: история очищена, "
                          "сеть вернулась к начальным весам.",
        "status_random": "Заданы случайные веса (равномерно от −0.5 до +0.5).",
        "status_standard": "Стандартный набор весов восстановлен, история очищена.",
        "status_history_cleared": "История потерь очищена, веса не изменены.",
        "status_no_history": "Истории потерь пока нет.",
        "status_stop_first": "Сначала нажмите «Стоп»: идёт автоматическое обучение.",
        "status_step": "Шаг {n} из {total}: {title}",
        "status_first_step": "Это первый шаг: впереди разбор примера.",
        "status_last_step": "Это последний шаг. Далее — следующий пример.",
        "status_sample": "Выбран пример {i} из {n}.",
        "status_weight_set": "{name} = {value}. История очищена: сеть изменилась вручную.",
        "status_full_mode": "Показан весь процесс: все {total} шагов для примера {i}.",
        "status_steps_reset": "Пошаговый режим возвращён к шагу 1.",

        "tab_detailed": "Пошагово",
        "tab_graph": "Потеря",
        "tab_pred": "Предсказ.",
        "tab_weights": "Веса",
        "cap_epoch": "Эпоха",
        "cap_sample": "Пример",
        "cap_eta": "Скорость η",
        "cap_correct": "Верных ответов",
        "lbl_init": "начало",
        "weights_note": "Поле — текущее значение параметра, «начало» — то, к чему вернёт Reset. Правка веса очищает историю: обучение начинается заново с новых весов.",
        "tab_compare": "До и после",
        "tab_help": "Справка",

        "graph_title": "Средняя потеря по эпохам",
        "graph_x": "Эпоха",
        "graph_y": "Среднее за эпоху",
        "layer_Inputs": "Входы",
        "layer_Hidden": "Скрытый слой",
        "layer_Output": "Выход",
        "dir_fwd": "прямой проход: слева направо",
        "dir_bwd": "обратный проход: справа налево",
        "legend_active": "янтарный = текущий шаг",
        "graph_points": "точек: {n}",
        "graph_empty": "Истории пока нет. Нажмите «Шаг эпоха» или «Обучение».",
        "graph_caption": "Одна точка = одна завершённая эпоха. Потеря может расти: "
                         "это стохастическое обновление, а не гарантированный спуск.",

        "pred_caption": "Предсказания текущей сети по всем примерам (веса не меняются).",
        "pred_note": "Ошибка = предсказание − цель. Порог классификации: 0.5.",
        "pred_legend": "Зелёным помечены примеры, где ответ сети совпал с целью. В колонке "
                       "«Сравнение»: предсказание до обучения → сейчас → цель.",

        "cmp_title": "Сравнение: до обучения и сейчас",
        "cmp_avg_before": "Средняя потеря до обучения",
        "cmp_avg_current": "Средняя потеря сейчас",
        "cmp_change": "Изменение средней потери",
        "cmp_row": "Пример {i}: {before} → {current} → цель {target}",
        "cmp_head": "Пример",
        "cmp_note": "Обе величины — оценочный прогон по датасету с неизменными весами: "
                    "до обучения и с текущими весами. Поэтому они сравнимы.",
        "cmp_correct": "Верных ответов: было {before} из {n}, стало {after} из {n}.",

        "det_head": "Шаг {n} из {total} · {title}",
        "det_mode_note": "Пошаговый режим считает математику по текущим весам, но НЕ записывает "
                         "обновления в сеть. Реальное обучение делают «Шаг эпоха» и «Обучение».",
        "det_full": "Весь процесс для примера {i}",
        "det_sample_line": "Пример {i} из {n}: x1 = {x1}, x2 = {x2}, цель = {y}",

        "stp1": "Выбор примера",
        "stp2": "Первый скрытый нейрон",
        "stp3": "Второй скрытый нейрон",
        "stp4": "Выход и предсказание",
        "stp5": "Потеря примера",
        "stp6": "Обратный проход: выход",
        "stp7": "Градиенты выходного слоя",
        "stp8": "Обратный проход: скрытый слой",
        "stp9": "Градиенты скрытого слоя",
        "stp10": "Обновление весов и следующий пример",

        "exp1": "Берём один пример из датасета: два числа на вход и цель, к которой "
                "надо приблизиться. Дальше вся математика — только про него.",
        "exp2": "Считаем взвешенную сумму входов первого скрытого нейрона и пропускаем "
                "её через сигмоиду. Сигмоида сжимает любое число в диапазон от 0 до 1.",
        "exp3": "То же самое для второго скрытого нейрона. Его веса отличаются, поэтому "
                "и ответ у него свой: два нейрона видят пример с двух сторон.",
        "exp4": "Скрытые значения становятся входами выходного нейрона. Его ответ σ(z_o) "
                "и есть предсказание сети — число от 0 до 1.",
        "exp5": "Сравниваем предсказание с целью: L = 1/2·(ŷ − y)². Множитель 1/2 нужен "
                "затем, чтобы при дифференцировании квадрата двойка сократилась.",
        "exp6": "Обратный проход начинается с конца: насколько сильно выходной сигнал "
                "влияет на потерю. Это дельта δ_o = ∂L/∂z_o.",
        "exp7": "Каждый вес выходного слоя получает свою производную: дельта выхода, "
                "умноженная на вход этого веса. Смещению нужен только сам вход — единица.",
        "exp8": "Теперь ошибка идёт дальше назад, в скрытый слой: сначала ∂L/∂h1 и ∂L/∂h2 "
                "через веса w5 и w6, затем умножение на σ'(z_h).",
        "exp9": "Аналогично выходному слою: дельта скрытого нейрона, умноженная на его "
                "вход. Так девять параметров получают девять градиентов.",
        "exp10": "Каждый параметр сдвигается против градиента: new = old − η·grad. "
                 "Дальше берётся следующий пример, а после последнего примера эпоха считается завершённой.",

        "why1": "Обучение идёт по одному примеру, поэтому важно не перепутать, какой пример "
                "и какие веса сейчас разбираются.",
        "why2": "Значение h1 — это то, что скрытый слой «понял» из примера. Именно оно пойдёт "
                "в выходной слой.",
        "why3": "Два скрытых нейрона дают сети два разных признака. При одинаковых весах они "
                "сливаются в один, и XOR становится неподъёмной.",
        "why4": "ŷ — единственное число, которое сеть выдаёт наружу. Всё остальное — способ его получить.",
        "why5": "Потеря — это мера, которую мы уменьшаем. Без неё не было бы направления обучения.",
        "why6": "Дельта — стартовая точка цепочки: все остальные производные выражаются через неё.",
        "why7": "Градиент говорит не только «насколько», но и «в какую сторону»: знак показывает, "
                "нужно ли вес увеличить или уменьшить.",
        "why8": "Скрытый слой не видит цель напрямую. Он узнаёт о своей ошибке от выходного слоя — "
                "в этом и смысл обратного распространения.",
        "why9": "Теперь у нас есть производные по всем девяти параметрам: по каждому видно, "
                "как сильно он виноват в ошибке этого примера.",
        "why10": "Один шаг почти ничего не меняет. Меняются многократные повторения: те же "
                 "примеры, те же формулы, уже другие веса.",

        "epoch_def": "Эпоха = один полный проход датасета. Для каждого примера: прямой проход, "
                     "потеря, обратный проход, градиенты, обновление всех девяти параметров. "
                     "Для датасета из четырёх примеров одна эпоха — это четыре обновления весов.",
        "eta_def": "η (эта) — размер шага. Градиент задаёт направление, η — длину шага.",
        "why_label": "Почему это важно:",

        "help_intro": "Десять разделов справки: от смысла обучения до влияния скорости обучения. "
                      "Формулы в обоих языках одинаковы.",
        "help_titles": ("Что такое обучение сети", "Датасет", "Прямой проход", "Предсказание",
                        "Потеря", "Обратное распространение", "Градиент", "Обновление весов",
                        "Эпоха", "Скорость обучения"),
        "help_b1": "Обучение — это многократное повторение одного цикла: сеть показывает пример, "
                   "сравнивает свой ответ с целью и чуть-чуть меняет свои веса так, чтобы ошибка "
                   "уменьшалась. Здесь этот цикл виден целиком и на числах, а не «внутри чёрного "
                   "ящика».\n\nСеть 2 → 2 → 1 имеет девять обучаемых параметров. Всё обучение — "
                   "это аккуратный подбор этих девяти чисел.",
        "help_b2": "В приложении три маленьких датасета: AND, OR и XOR. Каждый пример — два числа "
                   "на вход (0 или 1) и один ответ-цель (0 или 1).\n\nAND и OR линейно разделимы: "
                   "их умеет решить даже один сумматор с порогом. XOR — нет: чтобы её выучить, "
                   "нужен скрытый слой. Сравните эти три датасета между собой — это и есть "
                   "главный учебный вывод.",
        "help_b3": "Прямой проход считает сеть слева направо:\n"
                   "  z_h1 = x1·w1 + x2·w2 + b1,  h1 = σ(z_h1)\n"
                   "  z_h2 = x1·w3 + x2·w4 + b2,  h2 = σ(z_h2)\n"
                   "  z_o  = h1·w5 + h2·w6 + bo,  ŷ  = σ(z_o)\n\n"
                   "σ(z) = 1/(1 + e^(−z)) — гладкая функция, которая сжимает любое число в "
                   "интервал от 0 до 1. Благодаря ей сеть дифференцируема, то есть производные "
                   "существуют.",
        "help_b4": "Предсказание ŷ — число от 0 до 1. Для бинарной задачи ответ «да» считается "
                   "при ŷ ≥ 0.5. Важно: ŷ — это не уверенность и не вероятность из учебника, а "
                   "значение сигмоиды, которое мы интерпретируем как степень «да».\n\n"
                   "Вкладка «Предсказания» показывает ŷ по всем примерам сразу, поэтому видно, "
                   "как ответы подтягиваются к целям по мере обучения.",
        "help_b5": "Потеря одного примера: L = 1/2·(ŷ − y)². Она мала, когда ответ близок к цели, "
                   "и растёт вместе с расхождением.\n\nВ приложении два разных числа, которые "
                   "легко перепутать:\n"
                   "  • средняя потеря эпохи — среднее по примерам, посчитанное ПО ХОДУ эпохи, "
                   "когда веса уже меняются от примера к примеру;\n"
                   "  • средняя потеря при текущих весах — один оценочный прогон по всем примерам "
                   "с неизменными весами.\n\n"
                   "Первое число идёт на график, второе — в сравнение «до и после».",
        "help_b6": "Обратный проход — это цепочка производных, посчитанная справа налево. Сначала "
                   "выясняется, как ошибка зависит от выхода: δ_o = ∂L/∂z_o. Затем эта ошибка "
                   "распределяется по скрытым нейронам: ∂L/∂h1 = δ_o·w5 и ∂L/∂h2 = δ_o·w6, а "
                   "дальше умножается на σ'(z_h).\n\nСмысл: скрытый нейрон узнаёт о своей роли в "
                   "ошибке от тех весов, через которые он повлиял на выход.",
        "help_b7": "Градиент — производная потери по конкретному параметру. Он показывает, "
                   "насколько изменится потеря, если параметр чуть-чуть увеличить.\n\n"
                   "∂L/∂w5 = δ_o·h1, ∂L/∂w1 = δ_h1·x1, ∂L/∂b1 = δ_h1 и так далее — все девять "
                   "градиентов показаны числами для текущего примера.\n\n"
                   "Знак градиента важнее величины: положительный значит «уменьшить вес», "
                   "отрицательный — «увеличить».",
        "help_b8": "Обновление одно и то же для всех параметров:\n  новое = старое − η · градиент\n\n"
                   "Мы всегда идём против градиента, потому что он показывает направление роста "
                   "потери, а нам нужно её уменьшение. Вкладка «Пошагово» показывает до и после "
                   "для всех девяти параметров в одном месте.",
        "help_b9": "Одна эпоха = один полный проход датасета. Для четырёх примеров это четыре "
                   "обновления весов, и каждое — по своему примеру.\n\nПосле завершённой эпохи "
                   "считается средняя потеря и добавляется точкой на график. Поэтому «100 эпох» "
                   "для AND — это 400 обновлений весов.",
        "help_b10": "η — размер шага. Маленькое η обучение идёт медленно, но осторожно. "
                    "Умеренное η — обычно достаточно. Слишком большое η может раскачать сеть: "
                    "веса перескакивают минимум, потеря начинает расти.\n\n"
                    "Приложение ничего не обещает: подставьте η = 0.01, η = 0.5 и η = 1.5 и "
                    "посмотрите на график. Поведение сети вы увидите своё, а не заранее "
                    "нарисованное. Потеря не обязана уменьшаться на каждом шаге.",
    },
    "en": {
        "app_title": "Neural Network Training Lab",
        "subtitle": "How the network changes from sample to sample and epoch to epoch",
        "lang_ru": "RU",
        "lang_en": "EN",
        "author_note": "Author: SteveWorks001",

        "box_dataset": "Dataset",
        "box_sample": "Current sample",
        "box_params": "Training parameters",
        "box_actions": "Training controls",
        "box_weights": "Network parameters (editable)",
        "box_speed": "Training speed",
        "box_overview": "Training overview",

        "ds_AND": "AND",
        "ds_OR": "OR",
        "ds_XOR": "XOR (exclusive OR)",
        "desc_AND": "True only when both inputs are 1. Linearly separable.",
        "desc_OR": "True when at least one input is 1. Linearly separable.",
        "desc_XOR": "True when the inputs differ. NOT linearly separable: without a hidden "
                    "layer the network cannot learn it.",

        "th_sample": "Row",
        "th_x1": "x1",
        "th_x2": "x2",
        "th_target": "Target",
        "th_prediction": "Prediction",
        "th_error": "Error",
        "th_loss": "Loss",
        "th_param": "Parameter",
        "th_value": "Value",
        "th_gradient": "Gradient",
        "th_old": "Before",
        "th_new": "After",
        "th_change": "Change",
        "th_before": "Before training",
        "th_compare": "Compare",
        "th_current": "Current",

        "lbl_sample_of": "Sample {i} / {n}",
        "lbl_x1": "x1 = {v}",
        "lbl_x2": "x2 = {v}",
        "lbl_target": "target = {v}",
        "lbl_epochs": "Epochs:",
        "lbl_eta": "Learning rate η:",
        "lbl_epoch_of": "Epoch {e} / {total}",
        "lbl_status": "Status:",
        "lbl_epoch_avg": "Epoch mean loss",
        "lbl_eval_loss": "Loss (current weights)",
        "lbl_sample_loss": "Current sample loss",
        "lbl_correct": "Correct answers: {k} / {n}",
        "lbl_epochs_hint": "1–10000",
        "lbl_eta_hint": "0–1.5, comma is accepted",
        "lbl_dataset_note": "The current sample is highlighted in the table.",
        "lbl_epoch_def": "One epoch = every sample of the dataset in turn, with a weight update "
                         "after each sample.",
        "lbl_two_losses": "The epoch mean loss is measured while training runs (each sample sees "
                          "already updated weights), while the other number is a single evaluation "
                          "pass with fixed weights.",

        "btn_step_epoch": "Step Epoch",
        "btn_train": "Train",
        "btn_stop": "Stop",
        "btn_reset": "Reset",
        "btn_reset_history": "Reset History",
        "btn_random": "Randomize Weights",
        "btn_standard": "Standard Weights",
        "btn_prev": "Previous Step",
        "btn_next": "Next Step",
        "btn_reset_steps": "Reset Step",
        "btn_full": "Show Full Process",

        "speed_slow": "Slow",
        "speed_normal": "Normal",
        "speed_fast": "Fast",

        "grp_h1": "Hidden neuron 1",
        "grp_h2": "Hidden neuron 2",
        "grp_o": "Output neuron",
        "hint_w1": "x1 → h1",
        "hint_w2": "x2 → h1",
        "hint_b1": "bias of h1",
        "hint_w3": "x1 → h2",
        "hint_w4": "x2 → h2",
        "state_ready": "Ready",
        "state_training": "Training",
        "state_stopped": "Stopped",
        "state_completed": "Completed",
        "state_reset": "Reset",
        "hint_b2": "bias of h2",
        "hint_w5": "h1 → ŷ",
        "hint_w6": "h2 → ŷ",
        "hint_bo": "bias of ŷ",

        "status_ready": "Ready. Pick a dataset and start training.",
        "status_training": "Training…",
        "status_stopped": "Stopped at epoch {e}.",
        "status_completed": "Training completed: {total} epochs.",
        "status_reset": "Reset: the network is back to its initial parameters, history cleared.",
        "status_lang": "Language switched. Training, weights and history untouched.",
        "status_epoch_done": "Epoch {e}: mean loss {loss}.",
        "status_bad": "\"{raw}\" is not a number. The last valid value is kept.",
        "status_range": "{name}: allowed from {lo} to {hi}. Nearest allowed value taken.",
        "status_dataset": "Dataset: {name}. Training restarted from scratch: history "
                          "cleared, the network is back to its initial weights.",
        "status_random": "Random weights set (uniform from −0.5 to +0.5).",
        "status_standard": "Standard weights restored, history cleared.",
        "status_history_cleared": "Loss history cleared, weights unchanged.",
        "status_no_history": "There is no loss history yet.",
        "status_stop_first": "Press Stop first: automatic training is running.",
        "status_step": "Step {n} of {total}: {title}",
        "status_first_step": "This is the first step; the sample breakdown starts here.",
        "status_last_step": "This is the last step. Next comes the next sample.",
        "status_sample": "Sample {i} of {n} selected.",
        "status_weight_set": "{name} = {value}. History cleared: the network changed by hand.",
        "status_full_mode": "Full process shown: all {total} steps of sample {i}.",
        "status_steps_reset": "Step mode returned to step 1.",

        "tab_detailed": "Step mode",
        "tab_graph": "Loss",
        "layer_Inputs": "Inputs",
        "layer_Hidden": "Hidden layer",
        "layer_Output": "Output",
        "dir_fwd": "forward pass: left to right",
        "dir_bwd": "backward pass: right to left",
        "legend_active": "amber = current step",
        "graph_points": "points: {n}",
        "tab_pred": "Predictions",
        "tab_weights": "Weights",
        "cap_epoch": "Epoch",
        "cap_sample": "Sample",
        "cap_eta": "Learning rate η",
        "cap_correct": "Correct answers",
        "lbl_init": "start",
        "weights_note": "The field holds the current value; «start» is what Reset returns to. Editing a weight clears the history: training restarts from the new values.",
        "tab_compare": "Before / after",
        "tab_help": "Help",

        "graph_title": "Average Loss vs Epoch",
        "graph_x": "Epoch",
        "graph_y": "Epoch mean loss",
        "graph_empty": "No history yet. Press Step Epoch or Train.",
        "graph_caption": "One point = one completed epoch. Loss may rise: this is stochastic "
                         "updating, not a guaranteed descent.",

        "pred_caption": "Predictions of the current network for every sample (weights unchanged).",
        "pred_note": "Error = prediction − target. Classification threshold: 0.5.",
        "pred_legend": "Rows are green when the network answer matches the target. The Compare "
                       "column shows the prediction before training → now → target.",

        "cmp_title": "Comparison: before training and now",
        "cmp_avg_before": "Mean loss before training",
        "cmp_avg_current": "Mean loss now",
        "cmp_change": "Change of mean loss",
        "cmp_row": "Sample {i}: {before} → {current} → target {target}",
        "cmp_head": "Sample",
        "cmp_note": "Both numbers are evaluation passes over the dataset with fixed weights: one "
                    "before training, one with the current weights. That is why they are comparable.",
        "cmp_correct": "Correct answers: {before} of {n} before, {after} of {n} now.",

        "det_head": "Step {n} of {total} · {title}",
        "det_mode_note": "Step mode computes the mathematics from the current weights but does NOT "
                         "write the update back. Real training happens in Step Epoch and Train.",
        "det_full": "Full process for sample {i}",
        "det_sample_line": "Sample {i} of {n}: x1 = {x1}, x2 = {x2}, target = {y}",

        "stp1": "Select sample",
        "stp2": "Hidden neuron 1",
        "stp3": "Hidden neuron 2",
        "stp4": "Output and prediction",
        "stp5": "Sample loss",
        "stp6": "Backward pass: output",
        "stp7": "Output-layer gradients",
        "stp8": "Backward pass: hidden layer",
        "stp9": "Hidden-layer gradients",
        "stp10": "Weight update and next sample",

        "exp1": "We take one sample of the dataset: two input numbers and the target to approach. "
                "Everything below is about this single sample.",
        "exp2": "Compute the weighted input sum of the first hidden neuron and push it through the "
                "sigmoid, which squeezes any number into the range from 0 to 1.",
        "exp3": "The same for the second hidden neuron. Its weights differ, so its answer differs "
                "too: two neurons see the sample from two sides.",
        "exp4": "The hidden values become inputs of the output neuron. Its answer σ(z_o) is the "
                "network prediction, a number between 0 and 1.",
        "exp5": "Compare the prediction with the target: L = 1/2·(ŷ − y)². The 1/2 factor is there "
                "so that the 2 cancels when differentiating the square.",
        "exp6": "The backward pass starts at the end: how strongly the output influences the loss. "
                "That is the output delta δ_o = ∂L/∂z_o.",
        "exp7": "Each output weight gets its own derivative: the output delta times that weight's "
                "input. A bias needs only the input itself, which is 1.",
        "exp8": "Now the error travels further back into the hidden layer: first ∂L/∂h1 and ∂L/∂h2 "
                "through weights w5 and w6, then multiplied by σ'(z_h).",
        "exp9": "Same as for the output layer: the hidden delta times the neuron input. Now all nine "
                "parameters have gradients.",
        "exp10": "Every parameter moves against its gradient: new = old − η·grad. Then the next "
                 "sample is taken, and after the last sample the epoch is complete.",

        "why1": "Training runs one sample at a time, so it matters which sample and which weights "
                "are being examined.",
        "why2": "h1 is what the hidden layer extracted from the sample; it is what the output layer "
                "will receive.",
        "why3": "Two hidden neurons give the network two different features. With equal weights they "
                "collapse into one and XOR becomes impossible.",
        "why4": "ŷ is the only number the network produces; everything else is a way to obtain it.",
        "why5": "Loss is the quantity we minimize. Without it there is no direction to learn in.",
        "why6": "The delta is the starting point of the chain: every other derivative is expressed "
                "through it.",
        "why7": "A gradient gives not only \"how much\" but also \"which way\": its sign says whether "
                "the weight should shrink or grow.",
        "why8": "The hidden layer never sees the target. It learns about its own share of the error "
                "from the output layer — that is the point of backpropagation.",
        "why9": "Now all nine parameters have derivatives, so for each one we see how much it "
                "contributed to the error of this sample.",
        "why10": "A single step changes almost nothing. Repetition changes everything: same samples, "
                 "same formulas, already different weights.",

        "epoch_def": "One epoch = one full pass over the dataset. For each sample: forward pass, "
                     "loss, backward pass, gradients, update of all nine parameters. For a dataset "
                     "of four samples one epoch means four weight updates.",
        "eta_def": "η is the step size. The gradient sets the direction, η sets the length.",
        "why_label": "Why this matters:",

        "help_intro": "Ten help sections, from the meaning of training to the effect of the learning "
                      "rate. Formulas are identical in both languages.",
        "help_titles": ("What is neural-network training?", "Dataset", "Forward pass", "Prediction",
                        "Loss", "Backpropagation", "Gradient", "Weight update", "Epoch",
                        "Learning rate"),
        "help_b1": "Training repeats one cycle over and over: the network looks at a sample, "
                   "compares its answer with the target and nudges its weights so the error gets "
                   "smaller. Here the whole cycle is visible as real numbers instead of a black box.\n\n"
                   "The 2 → 2 → 1 network has nine trainable parameters. All of training is the "
                   "careful selection of those nine numbers.",
        "help_b2": "Three small datasets are built in: AND, OR and XOR. Each sample is two input "
                   "numbers (0 or 1) and one target (0 or 1).\n\nAND and OR are linearly separable "
                   "and a single threshold unit can solve them. XOR is not: learning it requires a "
                   "hidden layer. Comparing the three datasets is the main lesson of this lab.",
        "help_b3": "The forward pass walks through the network left to right:\n"
                   "  z_h1 = x1·w1 + x2·w2 + b1,  h1 = σ(z_h1)\n"
                   "  z_h2 = x1·w3 + x2·w4 + b2,  h2 = σ(z_h2)\n"
                   "  z_o  = h1·w5 + h2·w6 + bo,  ŷ  = σ(z_o)\n\n"
                   "σ(z) = 1/(1 + e^(−z)) is smooth and squeezes any number into 0…1, which is what "
                   "makes the network differentiable.",
        "help_b4": "The prediction ŷ is a number from 0 to 1. For a binary task we call ŷ ≥ 0.5 a "
                   "\"yes\". Note that ŷ is a sigmoid value we interpret as a degree of \"yes\", not "
                   "a probability taken from a textbook.\n\nThe Predictions tab shows ŷ for all "
                   "samples at once, so you can watch the answers move toward their targets.",
        "help_b5": "The loss of one sample is L = 1/2·(ŷ − y)²: small when the answer is close to "
                   "the target, growing with the mismatch.\n\nTwo different numbers are easy to "
                   "confuse:\n"
                   "  • epoch mean loss — averaged over samples WHILE the epoch runs, so each sample "
                   "sees weights that were already updated;\n"
                   "  • mean loss with current weights — a single evaluation pass with fixed weights.\n\n"
                   "The first one is plotted, the second one feeds the before/after comparison.",
        "help_b6": "The backward pass is a chain of derivatives computed right to left. First we find "
                   "how the error depends on the output: δ_o = ∂L/∂z_o. Then that error is spread "
                   "over the hidden neurons: ∂L/∂h1 = δ_o·w5 and ∂L/∂h2 = δ_o·w6, multiplied by "
                   "σ'(z_h).\n\nIn other words, a hidden neuron learns about its share of the error "
                   "through the weights it influenced the output with.",
        "help_b7": "A gradient is the derivative of the loss with respect to one parameter: it says "
                   "how the loss changes if that parameter grows a little.\n\n"
                   "∂L/∂w5 = δ_o·h1, ∂L/∂w1 = δ_h1·x1, ∂L/∂b1 = δ_h1 and so on — all nine gradients "
                   "are shown as numbers for the current sample.\n\nThe sign matters more than the "
                   "size: positive means \"shrink the weight\", negative means \"grow it\".",
        "help_b8": "One rule for every parameter:\n  new = old − η · gradient\n\nWe always move "
                   "against the gradient, because it points where the loss grows and we want it to "
                   "shrink. The Step-by-step tab shows before and after for all nine parameters in "
                   "one place.",
        "help_b9": "One epoch = one full pass over the dataset. For four samples that is four weight "
                   "updates, each driven by its own sample.\n\nAfter a completed epoch the mean loss "
                   "is computed and appended to the graph as one point. So \"100 epochs\" on AND "
                   "means 400 weight updates.",
        "help_b10": "η is the step size. A small η learns slowly but cautiously. A moderate η is "
                    "usually enough. A too-large η can destabilize the network: weights jump past the "
                    "minimum and the loss starts to grow.\n\nThe app promises nothing: try η = 0.01, "
                    "η = 0.5 and η = 1.5 and look at the graph. You will see the network's own "
                    "behaviour. Loss does not have to decrease at every step.",
    },
}

# ---------------------------------------------------------------
# 7. Пошаговый режим: этапы и текстовые блоки
# ---------------------------------------------------------------

def term_prod(a_value, b_value, a_fmt=None):
    """Член произведения: (значение, текст). Знак живёт в значении."""
    render = a_fmt or fmt
    return a_value * b_value, "%s·%s" % (render(a_value), fmt(abs(b_value)))


def term_value(v):
    """Член-одиночка (смещение)."""
    return v, fmt(abs(v))


def join_terms(*terms):
    """Сложение членов с явными знаками: «0·0.6000 + 0·0.4500 − 0.1000»."""
    out = []
    for i, (value, text) in enumerate(terms):
        if i == 0:
            out.append(MINUS + text if value < 0 else text)
        else:
            out.append(" " + (MINUS if value < 0 else "+") + " " + text)
    return "".join(out)


def inp(value):
    """Вход датасета (0 или 1) показываем как целое, а не как 0.0000."""
    return str(int(value)) if float(value).is_integer() else fmt(value)


def sub(label, expression, result):
    """Подстановка в две строки: формула с числами и её значение."""
    pad = " " * (len(label) + 3)
    return ["%s = %s" % (label, expression), "%s= %s" % (pad, fmt(result))]


# Что подсвечивать на схеме сети для каждого шага: фаза, узлы, рёбра.
STEP_FOCUS = {
    1: ("fwd", ("x1", "x2"), ()),
    2: ("fwd", ("h1",), (("x1", "h1"), ("x2", "h1"))),
    3: ("fwd", ("h2",), (("x1", "h2"), ("x2", "h2"))),
    4: ("fwd", ("o",), (("h1", "o"), ("h2", "o"))),
    5: ("fwd", ("L",), (("o", "L"),)),
    6: ("bwd", ("o", "L"), (("o", "L"),)),
    7: ("bwd", ("o",), (("h1", "o"), ("h2", "o"))),
    8: ("bwd", ("h1", "h2"), (("h1", "o"), ("h2", "o"))),
    9: ("bwd", ("h1", "h2"), (("x1", "h1"), ("x2", "h1"), ("x1", "h2"), ("x2", "h2"))),
    10: ("upd", ("x1", "x2", "h1", "h2", "o"),
         (("x1", "h1"), ("x2", "h1"), ("x1", "h2"), ("x2", "h2"), ("h1", "o"), ("h2", "o"))),
}


def _b1(s, ctx):
    """Выбор примера."""
    return ["x1 = %s      x2 = %s      target = %s" % (
        inp(s["x1"]), inp(s["x2"]), inp(s["y"])),
        "N = %d" % ctx["n_samples"],
        "",
        "%s: %s" % (T("box_dataset"), ctx["dataset_name"]),
        "\n".join("%s. x1=%s  x2=%s  y=%s" % (i + 1, inp(r[0]), inp(r[1]), inp(r[2]))
                  for i, r in enumerate(ctx["rows"]))]


def _b2(s, ctx):
    """Первый скрытый нейрон."""
    p, a = s["before"], s["act"]
    lines = [FORMULA["z_h1"]]
    lines += sub("z_h1", join_terms(term_prod(a["x1"], p["w1"], inp),
                                    term_prod(a["x2"], p["w2"], inp),
                                    term_value(p["b1"])), a["z_h1"])
    lines.append("")
    lines.append(FORMULA["h1"] + "      " + FORMULA["sigmoid"])
    lines += sub("h1", "σ(%s)" % fmt(a["z_h1"]), a["h1"])
    return lines


def _b3(s, ctx):
    """Второй скрытый нейрон."""
    p, a = s["before"], s["act"]
    lines = [FORMULA["z_h2"]]
    lines += sub("z_h2", join_terms(term_prod(a["x1"], p["w3"], inp),
                                    term_prod(a["x2"], p["w4"], inp),
                                    term_value(p["b2"])), a["z_h2"])
    lines.append("")
    lines.append(FORMULA["h2"])
    lines += sub("h2", "σ(%s)" % fmt(a["z_h2"]), a["h2"])
    return lines


def _b4(s, ctx):
    """Выход и предсказание."""
    p, a = s["before"], s["act"]
    lines = [FORMULA["z_o"]]
    lines += sub("z_o", join_terms(term_prod(a["h1"], p["w5"]),
                                   term_prod(a["h2"], p["w6"]),
                                   term_value(p["bo"])), a["z_o"])
    lines.append("")
    lines.append(FORMULA["y_hat"])
    lines += sub("ŷ", "σ(%s)" % fmt(a["z_o"]), a["y_hat"])
    lines.append("")
    lines.append("ŷ %s 0.5  →  %s = %s" % (
        "≥" if a["y_hat"] >= 0.5 else "<",
        T("th_prediction"), "1" if a["y_hat"] >= 0.5 else "0"))
    return lines


def _b5(s, ctx):
    """Потеря примера."""
    a, diff = s["act"], s["act"]["y_hat"] - s["y"]
    lines = [FORMULA["loss"]]
    lines += sub("L", "1/2·(%s)²" % (fmt(diff) if diff >= 0 else "(" + fmt(diff) + ")"), s["loss"])
    lines.append("")
    lines.append("|ŷ − y| = %s      L = %s" % (fmt(abs(diff)), fmt(s["loss"])))
    return lines


def _b6(s, ctx):
    """Обратный проход: выход."""
    a, b = s["act"], s["back"]
    lines = [FORMULA["d_loss_dy"]]
    lines += sub(d("ŷ"), "%s − %s" % (fmt(a["y_hat"]), inp(s["y"])), b["d_loss_d_y"])
    lines.append("")
    lines.append(FORMULA["d_y_dzo"] + "      " + FORMULA["sigmoid_prime"])
    lines += sub("∂ŷ/∂z_o", "%s·(1 − %s)" % (fmt(a["y_hat"]), fmt(a["y_hat"])), b["d_y_d_zo"])
    lines.append("")
    lines.append(FORMULA["delta_o"])
    lines += sub(d("z_o"), "%s · %s" % (fmt(b["d_loss_d_y"]), fmt(b["d_y_d_zo"])), b["delta_o"])
    lines.append("")
    lines.append("δ_o = " + fmt(b["delta_o"]))
    return lines


def _b7(s, ctx):
    """Градиенты выходного слоя."""
    a, b, g = s["act"], s["back"], s["back"]["grads"]
    lines = [FORMULA["dw5"]]
    lines += sub(d("w5"), "%s · %s" % (fmt(b["delta_o"]), fmt(a["h1"])), g["w5"])
    lines.append("")
    lines.append(FORMULA["dw6"])
    lines += sub(d("w6"), "%s · %s" % (fmt(b["delta_o"]), fmt(a["h2"])), g["w6"])
    lines.append("")
    lines.append(FORMULA["dbo"])
    lines += sub(d("bo"), fmt(b["delta_o"]), g["bo"])
    return lines


def _b8(s, ctx):
    """Обратный проход: скрытый слой."""
    a, b, p = s["act"], s["back"], s["before"]
    lines = [FORMULA["dh1"]]
    lines += sub(d("h1"), "%s · %s" % (fmt(b["delta_o"]), fmt(p["w5"])), b["d_loss_d_h1"])
    lines.append("")
    lines.append(FORMULA["dh2"])
    lines += sub(d("h2"), "%s · %s" % (fmt(b["delta_o"]), fmt(p["w6"])), b["d_loss_d_h2"])
    lines.append("")
    lines.append(FORMULA["sigmoid_prime"])
    lines += sub("σ'(z_h1)", "%s·(1 − %s)" % (fmt(a["h1"]), fmt(a["h1"])), b["d_h1_d_zh1"])
    lines.append("")
    lines += sub("σ'(z_h2)", "%s·(1 − %s)" % (fmt(a["h2"]), fmt(a["h2"])), b["d_h2_d_zh2"])
    lines.append("")
    lines.append(FORMULA["dz_h1"])
    lines += sub(d("z_h1"), "%s · %s" % (fmt(b["d_loss_d_h1"]), fmt(b["d_h1_d_zh1"])), b["delta_h1"])
    lines.append("")
    lines.append(FORMULA["dz_h2"])
    lines += sub(d("z_h2"), "%s · %s" % (fmt(b["d_loss_d_h2"]), fmt(b["d_h2_d_zh2"])), b["delta_h2"])
    return lines


def _b9(s, ctx):
    """Градиенты скрытого слоя: шесть оставшихся параметров."""
    a, b, g = s["act"], s["back"], s["back"]["grads"]
    delta = {"h1": b["delta_h1"], "h2": b["delta_h2"]}
    rows = (("w1", "h1", a["x1"]), ("w2", "h1", a["x2"]), ("b1", "h1", None),
            ("w3", "h2", a["x1"]), ("w4", "h2", a["x2"]), ("b2", "h2", None))
    lines = []
    for key, neuron, factor in rows:
        lines.append(FORMULA["d" + key])
        if factor is None:
            lines += sub(d(key), fmt(delta[neuron]), g[key])
        else:
            lines += sub(d(key), "%s · %s" % (fmt(delta[neuron]), fmt(factor)), g[key])
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _b10(s, ctx):
    """Обновление девяти параметров и переход к следующему примеру."""
    g, eta = s["back"]["grads"], s["eta"]
    before, after = s["before"], s["after"]
    lines = [FORMULA["update"], "η = " + fmt(eta), ""]
    for key in PARAM_KEYS:
        lines.append("%s_new = %s_old − η · %s" % (key, key, d(key)))
        lines.append("       = %s − %s · (%s) = %s" % (
            fmt(before[key]), fmt(eta), fmt(g[key]), fmt(after[key])))
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")
    lines.append("→ " + T("lbl_sample_of", i=ctx["next_sample"], n=ctx["n_samples"]))
    return lines


STEP_BLOCKS = {1: _b1, 2: _b2, 3: _b3, 4: _b4, 5: _b5,
               6: _b6, 7: _b7, 8: _b8, 9: _b9, 10: _b10}


def step_head(n):
    return T("det_head", n=n, total=TOTAL_STEPS, title=T("stp%d" % n))


def render_step(n, s, ctx):
    """Один шаг: объяснение, формула, подстановка чисел, результат, «зачем»."""
    parts = ["=" * 66,
             step_head(n),
             "=" * 66,
             "",
             T("exp%d" % n),
             "",
             "\n".join(STEP_BLOCKS[n](s, ctx)),
             "",
             T("why_label") + " " + T("why%d" % n)]
    return "\n".join(parts)


def render_all(s, ctx):
    """Весь процесс для примера сразу: все десять шагов одним текстом."""
    blocks = [T("det_full", i=ctx["sample_no"])]
    blocks += [render_step(n, s, ctx) for n in range(1, TOTAL_STEPS + 1)]
    return ("\n\n" + "—" * 33 + "\n\n").join(blocks)


# ---------------------------------------------------------------
# 9. Виджеты: поле параметра, схема сети, график потерь, прокрутка
# ---------------------------------------------------------------

class ScrollFrame(ttk.Frame):
    """Вертикальная прокрутка: левая колонка местами выше окна."""

    def __init__(self, master, width=330, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=COL_BG, width=width)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bind("<Enter>", self._grab_wheel)
        self.bind("<Leave>", self._release_wheel)

    def _on_inner(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        # Ширина содержимого равна ширине окна: иначе прокрутка «гуляет» по горизонтали.
        self.canvas.itemconfigure(self._win, width=event.width)

    def _wheel(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _grab_wheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)

    def _release_wheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")


def to_text(value, integer=False):
    """Как показать число в поле ввода: только ASCII, минус обычный.
    Целочисленное значение вещественного поля остаётся с «.0», чтобы
    зажатая граница (12.5 -> 5.0) выглядела в поле ровно как число сети."""
    if integer:
        return str(int(round(value)))
    text = ("%.4f" % value).rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


class ParamField(ttk.Frame):
    """Числовое поле с мягкой валидацией.

    Пустой ввод, буквы, запятая вместо точки, выход за диапазон — приложение
    не падает: поле подсвечивается, прежнее значение сети остаётся.
    """

    def __init__(self, master, key, value, lo, hi, on_change, fonts,
                 symbol_key=None, hint_key=None, integer=False, label_width=6):
        super().__init__(master)
        self.key = key
        self.lo, self.hi = lo, hi
        self.integer = integer
        self.on_change = on_change
        self.fonts = fonts
        self._lock = False
        self._bad = False
        parts = []
        if symbol_key:
            self.symbol = ttk.Label(self, text=symbol_key, width=label_width, anchor="w")
            self.symbol.pack(side="left")
            parts.append(self.symbol)
        if hint_key:
            self.hint_key = hint_key
            self.hint = ttk.Label(self, text=T(hint_key), anchor="w",
                                  foreground="#6a6a63", font=fonts["tiny"])
            self._fit_hint()
            self.hint.pack(side="left")
            parts.append(self.hint)
        self.var = tk.StringVar(value=to_text(value, integer))
        self.entry = ttk.Entry(self, textvariable=self.var, width=10, justify="right",
                               font=fonts["small"])
        self.entry.pack(side="right")
        self.var.trace_add("write", self._changed)

    def _mark(self, bad):
        if bad == self._bad:
            return
        self._bad = bad
        self.entry.configure(style="BadEntry.TEntry" if bad else "TEntry")

    def _changed(self, *_args):
        if self._lock:
            return
        raw = self.var.get()
        value = parse_number(raw)
        if value is None:
            self._mark(True)
            self.on_change(None, raw, self.key)
            return
        if value < self.lo or value > self.hi:
            value = clamp(value, self.lo, self.hi)
            if self.integer:
                value = round(value)
            self._show_clamped(value)
            self._mark(True)
            self.on_change(value, raw, self.key, True)
            return
        self._mark(False)
        self.on_change(int(round(value)) if self.integer else value, raw, self.key)

    def _show_clamped(self, value):
        """Показываем то число, которое реально попало в сеть, а не отказ."""
        self._lock = True
        self.var.set(to_text(value, self.integer))
        self._lock = False

    def set_value(self, value):
        self._lock = True
        self.var.set(to_text(value, self.integer))
        self._lock = False
        self._mark(False)

    def _fit_hint(self):
        """Ширина подсказки в символах считается по настоящему замеру текста:
        кириллица шире «0», поэтому фиксированные 14 символов резали «смещение h1»."""
        unit = self.fonts["tiny"].measure("0") or 1
        need = self.fonts["tiny"].measure(str(self.hint.cget("text"))) + 6
        self.hint.configure(width=max(int(math.ceil(need / float(unit))), 8))

    def retranslate(self):
        if hasattr(self, "hint"):
            self.hint.configure(text=T(self.hint_key))
            self._fit_hint()


class NetworkView(ttk.Frame):
    """Схема 2 → 2 → 1 на tk.Canvas: веса на связях, значения в узлах."""

    R = 26
    LOSS_W = 46
    LOSS_H = 40         # 21px на «L» + 17px на значение, без наложения
    # (доля длины ребра, вертикальный сдвиг): метки разносятся так, чтобы
    # скрещивающиеся рёбра x1→h2 и x2→h1 не подписывались в одной точке.
    EDGE_LABEL_AT = {
        ("x1", "h1"): (0.42, -13),
        ("x2", "h1"): (0.66, -14),
        ("x1", "h2"): (0.66, 14),
        ("x2", "h2"): (0.42, 14),
        ("h1", "o"): (0.56, -15),
        ("h2", "o"): (0.56, 15),
    }
    BIAS_LABEL_AT = {
        "b1": ("h1", 28, -28),
        "b2": ("h2", 28, 32),
        "bo": ("o", 28, -30),
    }

    def __init__(self, master, app, fonts, **kw):
        super().__init__(master, **kw)
        self.app = app
        self.fonts = fonts
        self.canvas = tk.Canvas(self, bg=COL_PAPER, highlightthickness=1,
                                highlightbackground=COL_FRAME, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

    # -- низ-level ---------------------------------------------------------
    def _text(self, c, x, y, s, font=None, fill=COL_TEXT, **kw):
        return c.create_text(x, y, text=s, font=font or self.fonts["small"],
                             fill=fill, **kw)

    def _arrow(self, c, x1, y1, x2, y2, color, width=2, dash=None):
        c.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash, capstyle="round")
        angle = math.atan2(y2 - y1, x2 - x1)
        for turn in (152, -152):
            a = angle + math.radians(turn)
            c.create_line(x2, y2, x2 + 11 * math.cos(a), y2 + 11 * math.sin(a),
                          fill=color, width=width, capstyle="round")

    def _node(self, c, x, y, name, value, active, sub_text=""):
        c.create_oval(x - self.R, y - self.R, x + self.R, y + self.R,
                      fill=COL_ACTIVE_FILL if active else COL_NODE,
                      outline=COL_ACTIVE if active else COL_NODE_LINE,
                      width=3 if active else 1)
        # Имя и значение держим внутри круга с запасом: при шрифтах mid/small
        # они наезжали друг на друга (замерено bbox на canvas).
        self._text(c, x, y - 10, name, self.fonts["smallb"])
        if value:
            self._text(c, x, y + 11, value, self.fonts["small"])
        if sub_text:
            self._text(c, x, y + self.R + 11, sub_text, self.fonts["tiny"], fill="#5a5a53")

    # -- отрисовка ---------------------------------------------------------
    def redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 420), max(c.winfo_height(), 300)
        trainer = self.app.trainer
        p = trainer.params
        # Значения в узлах — всегда про текущий пример и текущие веса.
        x1, x2, _y = trainer.current_sample
        act = forward(x1, x2, p)
        phase, nodes, edges = self.app.step_focus()
        back = phase == "bwd"
        compact = w < 520

        top, bottom = 44, h - 34
        x_x = 66
        x_h = 66 + (w - 150) * (0.42 if not compact else 0.45)
        x_o = 66 + (w - 150) * (0.78 if not compact else 0.82)
        pos = {
            "x1": (x_x, top + (bottom - top) * 0.30),
            "x2": (x_x, top + (bottom - top) * 0.72),
            "h1": (x_h, top + (bottom - top) * 0.26),
            "h2": (x_h, top + (bottom - top) * 0.76),
            "o": (x_o, top + (bottom - top) * 0.51),
        }
        lx = min(w - 44, x_o + 96)
        ly = pos["o"][1]
        pos["L"] = (lx, ly)

        # Связи: прямые — сплошные, обратные — пунктир справа налево.
        for (src, dst), key in EDGE_PARAM.items():
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            active = (src, dst) in edges
            if back and not active:
                color, width, dash = "#7fa0ba", 1, None
            elif active:
                color, width, dash = (COL_BWD if back else COL_ACTIVE), 3, None
            else:
                color, width, dash = COL_FWD, 2, None
            dx, dy = x2 - x1, y2 - y1
            norm = math.hypot(dx, dy) or 1.0
            sx = x1 + dx * self.R / norm
            sy = y1 + dy * self.R / norm
            ex = x2 - dx * self.R / norm
            ey = y2 - dy * self.R / norm
            self._arrow(c, sx, sy, ex, ey, color, width, dash)
            t, off = self.EDGE_LABEL_AT[(src, dst)]
            self._text(c, x1 + dx * t, y1 + dy * t + off,
                       "%s=%s" % (key, fmt(p[key], 3)), self.fonts["smallb"],
                       fill=COL_ACTIVE if active else COL_TEXT)

        # Смещения показаны у своих нейронов: отдельный вход «1» только путает.
        # Справа от узла (сверху для h1 и выхода, снизу для h2) — там нет ни рёбер,
        # ни подписей весов, поэтому число не наезжает на линии.
        for key, (node, dx_off, dy_off) in self.BIAS_LABEL_AT.items():
            x, y = pos[node]
            self._text(c, x + dx_off, y + dy_off, "%s = %s" % (key, fmt(p[key], 3)),
                       self.fonts["tiny"], anchor="w",
                       fill=COL_ACTIVE if node in nodes else "#5a5a53")

        values = {
            "x1": inp(act["x1"]), "x2": inp(act["x2"]),
            "h1": fmt(act["h1"], 3), "h2": fmt(act["h2"], 3),
            "o": fmt(act["y_hat"], 3),
        }
        names = {"x1": "x1", "x2": "x2", "h1": "h1", "h2": "h2", "o": "ŷ"}
        for key, (x, y) in pos.items():
            if key == "L":
                continue
            self._node(c, x, y, names[key], values[key], key in nodes)

        # Блок потери: L = 1/2·(ŷ − y)² для текущего примера.
        x, y = pos["L"]
        loss_active = "L" in nodes
        c.create_rectangle(x - self.LOSS_W / 2, y - self.LOSS_H / 2,
                           x + self.LOSS_W / 2, y + self.LOSS_H / 2,
                           fill=COL_ACTIVE_FILL if loss_active else COL_NODE,
                           outline=COL_ACTIVE if loss_active else COL_NODE_LINE,
                           width=3 if loss_active else 1)
        self._text(c, x, y - 9, "L", self.fonts["smallb"])
        loss = sample_loss(act["y_hat"], trainer.current_sample[2])
        self._text(c, x, y + 11, fmt(loss, 3), self.fonts["tiny"])
        self._arrow(c, pos["o"][0] + self.R, pos["o"][1],
                    x - self.LOSS_W / 2 - 2, y,
                    COL_ACTIVE if loss_active else COL_DIM, 2, dash=(4, 3) if back else None)

        # Подписи слоёв и легенда направления расчёта.
        for label, x in (("Inputs", pos["x1"][0]), ("Hidden", pos["h1"][0]), ("Output", pos["o"][0])):
            self._text(c, x, 16, T("layer_" + label), self.fonts["smallb"], fill="#5a5a53")
        hint = T("dir_bwd") if back else T("dir_fwd")
        color = COL_BWD if back else COL_FWD
        self._text(c, 12, h - 12, hint, self.fonts["tiny"], fill=color, anchor="w")
        self._text(c, w - 12, h - 12, T("legend_active"), self.fonts["tiny"],
                   fill=COL_ACTIVE, anchor="e")


def nice_ceiling(value):
    """Верхняя граница оси Y «красивым» числом: 0.05, 0.1, 0.25, 0.5, 1.0 …"""
    if value <= 0:
        return 0.1
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= candidate * base:
            return candidate * base
    return 10.0 * base


class LossGraph(ttk.Frame):
    """График «средняя потеря по эпохам» на tk.Canvas, без matplotlib."""

    PAD_L = 70          # слева место под повёрнутую подпись оси и деления 0.0750
    PAD_R = 14
    PAD_T = 52          # полоса заголовка + строка «точек / последнее значение»
    PAD_B = 52          # снизу деления эпох, и под ними отдельная подпись оси
    INFO_Y = 35         # сюда же, под заголовок, выводятся служебные числа
    MAX_SEGMENTS = 500   # при десятках тысяч эпох рисуем с шагом, иначе тормозит

    def __init__(self, master, app, fonts, **kw):
        super().__init__(master, **kw)
        self.app = app
        self.fonts = fonts
        self.canvas = tk.Canvas(self, bg=COL_PAPER, highlightthickness=1,
                                highlightbackground=COL_FRAME, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 360), max(c.winfo_height(), 220)
        c.create_text(w / 2.0, 12, text=T("graph_title"), font=self.fonts["midb"], fill=COL_TEXT)
        history = self.app.trainer.history
        if not history:
            c.create_text(w / 2.0, h / 2.0, text=T("graph_empty"), font=self.fonts["small"],
                          fill=COL_DIM, width=w - 60, justify="center")
            return
        left, right = self.PAD_L, w - self.PAD_R
        top, bottom = self.PAD_T, h - self.PAD_B
        epochs = [e for e, _ in history]
        losses = [l for _, l in history]
        y_max = nice_ceiling(max(losses))
        x_max = max(epochs)
        x_min = 1 if x_max > 1 else 0

        def px(epoch):
            if x_max == x_min:
                return (left + right) / 2.0
            return left + (right - left) * (epoch - x_min) / float(x_max - x_min)

        def py(loss):
            return bottom - (bottom - top) * (loss / y_max)

        # Сетка и подписи осей.
        for i in range(5):
            value = y_max * i / 4.0
            y = py(value)
            c.create_line(left, y, right, y, fill=COL_GRID)
            c.create_text(left - 6, y, text=fmt(value, 4), font=self.fonts["tiny"],
                          fill="#5a5a53", anchor="e")
        ticks = min(6, x_max - x_min + 1)
        for i in range(ticks):
            epoch = x_min + int(round(i * (x_max - x_min) / float(max(ticks - 1, 1))))
            x = px(epoch)
            c.create_line(x, bottom, x, bottom + 4, fill=COL_NODE_LINE)
            c.create_text(x, bottom + 14, text=str(epoch), font=self.fonts["tiny"],
                          fill="#5a5a53")
        c.create_line(left, top, left, bottom, fill=COL_NODE_LINE)
        c.create_line(left, bottom, right, bottom, fill=COL_NODE_LINE)
        c.create_text((left + right) / 2.0, h - 12, text=T("graph_x"),
                      font=self.fonts["small"], fill="#5a5a53")
        c.create_text(14, (top + bottom) / 2.0, text=T("graph_y"), font=self.fonts["small"],
                      fill="#5a5a53", angle=90)

        # Кривая: все точки, но не больше MAX_SEGMENTS отрезков.
        stride = max(1, int(math.ceil(len(history) / float(self.MAX_SEGMENTS))))
        pts = [(px(e), py(l)) for e, l in history[::stride]]
        if pts[-1] != (px(history[-1][0]), py(history[-1][1])):
            pts.append((px(history[-1][0]), py(history[-1][1])))
        if len(pts) > 1:
            c.create_line(*[v for pt in pts for v in pt], fill=COL_CURVE, width=2)
        for x, y in pts:
            c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5,
                          fill=COL_CURVE_PT if (x, y) == pts[-1] else COL_CURVE, outline="")
        last = history[-1]
        c.create_text(right - 6, self.INFO_Y,
                      text="%s: %s" % (T("lbl_epoch_avg"), fmt(last[1], 4)),
                      font=self.fonts["tiny"], fill=COL_ACTIVE, anchor="e")
        c.create_text(left + 6, self.INFO_Y, text=T("graph_points", n=len(history)),
                      font=self.fonts["tiny"], fill="#5a5a53", anchor="w")


# ---------------------------------------------------------------
# 10. Главное окно приложения
# ---------------------------------------------------------------

# Короткое название состояния для обзора; длинные сообщения — в строке статуса.
STATE_KEY = {
    "ready": "state_ready",
    "training": "state_training",
    "stopped": "state_stopped",
    "completed": "state_completed",
    "reset": "state_reset",
}


class TrainingLabApp(tk.Tk):
    """Три колонки: слева данные и управление, в центре схема и обзор,
    справа пять вкладок: пошаговый режим, график, предсказания, веса, справка."""

    def __init__(self):
        super().__init__()
        self.trainer = Trainer("XOR", dict(STANDARD_PARAMS), 0.1, 100)
        self.step_no = 1
        self.full_mode = False
        self.help_index = 0
        self._after_id = None
        self._syncing_table = False
        self._text_refs = []
        self._radios = []
        self._tabs = []

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d" % (min(1520, sw - 40), min(940, sh - 60)))
        # Минимум = наименьший из поддерживаемых размеров, иначе окно 1366x768
        # молча поднималось бы до 780 и честная проверка размера была бы невозможна.
        self.minsize(min(1300, sw - 40), min(768, sh - 60))
        self.title(T("app_title"))
        self.configure(bg=COL_BG)

        self._make_fonts()
        self._make_styles()
        self._build_header()
        self._build_left()
        self._build_center()
        self._build_right()
        self._build_bottom()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh()
        self.set_status("status_ready")

    # -- оформление --------------------------------------------------------
    def _make_fonts(self):
        base = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")
        self.fonts = {
            "title": tkfont.Font(family=base, size=17, weight="bold"),
            "sub": tkfont.Font(family=base, size=14),
            "mid": tkfont.Font(family=base, size=15, weight="bold"),
            "midb": tkfont.Font(family=base, size=13, weight="bold"),
            "small": tkfont.Font(family=base, size=12),
            "smallb": tkfont.Font(family=base, size=12, weight="bold"),
            "tiny": tkfont.Font(family=base, size=10),
            "mono": tkfont.Font(family=mono, size=12),
            "monob": tkfont.Font(family=mono, size=12, weight="bold"),
        }
        self.option_add("*Font", self.fonts["sub"])

    def _make_styles(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=COL_BG, foreground=COL_TEXT)
        st.configure("TFrame", background=COL_BG)
        st.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        st.configure("TLabelframe", background=COL_BG, bordercolor=COL_FRAME)
        st.configure("TLabelframe.Label", background=COL_BG, foreground="#3a3a35",
                     font=self.fonts["midb"])
        st.configure("TNotebook", background=COL_BG, bordercolor=COL_FRAME)
        # Вкладок пять, и они обязаны помещаться в 1300px: шрифт и отступы держим компактнее.
        st.configure("TNotebook.Tab", padding=(10, 5), font=self.fonts["small"])
        st.configure("TButton", padding=(10, 6), font=self.fonts["sub"])
        st.configure("Go.TButton", font=self.fonts["midb"])
        st.configure("TRadiobutton", background=COL_BG, font=self.fonts["sub"])
        st.configure("Treeview", background=COL_PAPER, fieldbackground=COL_PAPER,
                     font=self.fonts["small"], rowheight=24)
        st.configure("Treeview.Heading", font=self.fonts["smallb"])
        # В таблице предсказаний 7 колонок: заголовки держим узким шрифтом,
        # иначе на 1300px они обрезаются.
        st.configure("Pred.Treeview", background=COL_PAPER, fieldbackground=COL_PAPER,
                     font=self.fonts["small"], rowheight=24)
        st.configure("Pred.Treeview.Heading", font=self.fonts["tiny"])
        st.map("TNotebook.Tab", background=[("selected", COL_PAPER)],
               foreground=[("selected", COL_TEXT)])
        # Имя стиля для «плохого» поля обязано быть вида X.TEntry, иначе ttk его игнорирует.
        st.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG, foreground=COL_BAD_FG)

    def _reg(self, widget, key, attr="text", **kw):
        widget.configure(**{attr: T(key, **kw)})
        self._text_refs.append((widget, key, attr, kw))
        return widget

    def _box(self, title_key, padding=(8, 6)):
        box = ttk.LabelFrame(self.left_inner, text=T(title_key), padding=padding)
        box.pack(fill="x", pady=(0, 8))
        self._text_refs.append((box, title_key, "text", {}))
        return box

    # -- верхняя строка ----------------------------------------------------
    def _build_header(self):
        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 2))
        self._reg(ttk.Label(head, text=T("app_title"), font=self.fonts["title"]),
                  "app_title", font=self.fonts["title"]).pack(side="left")
        self._reg(ttk.Label(head, text=T("subtitle"), font=self.fonts["small"],
                            foreground="#4a4a4a"), "subtitle",
                  font=self.fonts["small"]).pack(side="left", padx=(16, 0))
        lang = ttk.Frame(head)
        lang.pack(side="right")
        self.var_lang = tk.StringVar(value=LANG)
        for code in ("ru", "en"):
            r = ttk.Radiobutton(lang, text=T("lang_" + code), value=code, variable=self.var_lang,
                                command=lambda c=code: self.switch_language(c))
            r.pack(side="left", padx=(0, 6))
            self._radios.append((r, "lang_" + code, code))

    # -- левая колонка -----------------------------------------------------
    def _build_left(self):
        scroll = ScrollFrame(self, width=310)
        scroll.grid(row=1, column=0, sticky="ns", padx=(12, 0), pady=4)
        self.left_inner = scroll.inner
        # Центру фиксировано 470px: ниже этого схема сети перестаёт читаться,
        # а прирастает окно — правая колонка с таблицами и графиком.
        self.columnconfigure(0, minsize=314, weight=0)
        self.columnconfigure(1, minsize=470, weight=0)
        self.columnconfigure(2, minsize=430, weight=1)
        self.rowconfigure(1, weight=1)

        # Датасет
        box = self._box("box_dataset")
        self.var_dataset = tk.StringVar(value=self.trainer.dataset_id)
        for name in DATASET_IDS:
            r = ttk.Radiobutton(box, text=T("ds_" + name), value=name,
                                variable=self.var_dataset, command=self.on_dataset)
            r.pack(anchor="w")
            self._radios.append((r, "ds_" + name, None))
        self.desc_label = self._reg(ttk.Label(box, text=T("desc_" + self.trainer.dataset_id),
                                              font=self.fonts["tiny"], wraplength=284,
                                              justify="left"),
                                    "desc_" + self.trainer.dataset_id, font=self.fonts["tiny"])
        self.desc_label.pack(anchor="w", pady=(4, 2))

        cols = ("n", "x1", "x2", "y")
        self.ds_table = ttk.Treeview(box, columns=cols, show="headings", height=5)
        for col in cols:
            self.ds_table.heading(col, text=T("th_" + {"n": "sample", "y": "target"}.get(col, col)))
            self.ds_table.column(col, width=62, anchor="center")
        self.ds_table.pack(fill="x", pady=(2, 2))
        self.ds_table.tag_configure("current", background=COL_ACTIVE_FILL,
                                    font=self.fonts["smallb"])
        self.ds_table.bind("<<TreeviewSelect>>", self.on_sample_select)
        self._reg(ttk.Label(box, text=T("lbl_dataset_note"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=284, justify="left"),
                  "lbl_dataset_note", font=self.fonts["tiny"]).pack(anchor="w")

        # Текущий пример
        box = self._box("box_sample", padding=(8, 4))
        self.var_sample_of = tk.StringVar()
        ttk.Label(box, textvariable=self.var_sample_of, font=self.fonts["smallb"]).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w")
        self.var_x1 = tk.StringVar()
        self.var_x2 = tk.StringVar()
        self.var_y = tk.StringVar()
        for var in (self.var_x1, self.var_x2, self.var_y):
            ttk.Label(row, textvariable=var, font=self.fonts["small"]).pack(side="left", padx=(0, 10))

        # Параметры обучения
        box = self._box("box_params")
        self.field_epochs = ParamField(box, "epochs", self.trainer.total_epochs,
                                       *LIMITS["epochs"], on_change=self.on_epochs,
                                       fonts=self.fonts, symbol_key=T("lbl_epochs"), integer=True,
                                       label_width=9)
        self.field_epochs.pack(fill="x", pady=1)
        self.field_eta = ParamField(box, "eta", self.trainer.eta, *LIMITS["eta"],
                                    on_change=self.on_eta, fonts=self.fonts,
                                    symbol_key="η =", label_width=9)
        self.field_eta.pack(fill="x", pady=1)
        self._reg(ttk.Label(box, text=T("lbl_eta_hint"), font=self.fonts["tiny"],
                            foreground="#5a5a53"), "lbl_eta_hint",
                  font=self.fonts["tiny"]).pack(anchor="w", pady=(2, 0))
        self._reg(ttk.Label(box, text=T("lbl_epoch_def"), font=self.fonts["tiny"],
                            wraplength=284, justify="left", foreground="#5a5a53"),
                  "lbl_epoch_def", font=self.fonts["tiny"]).pack(anchor="w", pady=(4, 0))

        # Скорость
        box = self._box("box_speed", padding=(8, 4))
        self.var_speed = tk.StringVar(value="normal")
        for code in ("slow", "normal", "fast"):
            r = ttk.Radiobutton(box, text=T("speed_" + code), value=code, variable=self.var_speed)
            r.pack(anchor="w")
            self._radios.append((r, "speed_" + code, None))

        # Управление
        box = self._box("box_actions", padding=(8, 4))
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        self.btn_step = ttk.Button(grid, text=T("btn_step_epoch"), style="Go.TButton",
                                   command=self.step_epoch)
        self.btn_step.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.btn_train = ttk.Button(grid, text=T("btn_train"), style="Go.TButton",
                                    command=self.start_training)
        self.btn_train.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        self.btn_stop = ttk.Button(grid, text=T("btn_stop"), command=self.stop_training)
        self.btn_stop.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.btn_reset = ttk.Button(grid, text=T("btn_reset"), command=self.reset)
        self.btn_reset.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        self.btn_history = ttk.Button(grid, text=T("btn_reset_history"), command=self.reset_history)
        self.btn_history.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        for b, key in ((self.btn_step, "btn_step_epoch"), (self.btn_train, "btn_train"),
                       (self.btn_stop, "btn_stop"), (self.btn_reset, "btn_reset"),
                       (self.btn_history, "btn_reset_history")):
            self._text_refs.append((b, key, "text", {}))
        self._set_training_buttons(False)

    # -- центральная колонка ----------------------------------------------
    def _build_center(self):
        center = ttk.Frame(self)
        center.grid(row=1, column=1, sticky="nsew", padx=(6, 6), pady=4)

        overview = ttk.LabelFrame(center, text=T("box_overview"), padding=(10, 6))
        overview.pack(side="top", fill="x")
        self._text_refs.append((overview, "box_overview", "text", {}))
        self.var_epoch = tk.StringVar()
        self.var_eta_show = tk.StringVar()
        self.var_epoch_avg = tk.StringVar()
        self.var_eval = tk.StringVar()
        self.var_sample_loss = tk.StringVar()
        self.var_correct = tk.StringVar()
        self.var_status_line = tk.StringVar()
        self.var_before = tk.StringVar()
        self.var_now = tk.StringVar()
        cells = (("cap_epoch", self.var_epoch), ("cap_sample", self.var_sample_of),
                 ("cap_eta", self.var_eta_show),
                 ("lbl_epoch_avg", self.var_epoch_avg),
                 ("lbl_eval_loss", self.var_eval), ("lbl_sample_loss", self.var_sample_loss),
                 ("cap_correct", self.var_correct), ("lbl_status", self.var_status_line))
        for i, (key, var) in enumerate(cells):
            r, col = divmod(i, 3)
            cell = ttk.Frame(overview)
            cell.grid(row=r, column=col, sticky="w", padx=(0, 12))
            lab = ttk.Label(cell, text=T(key), font=self.fonts["tiny"], foreground="#5a5a53")
            lab.grid(row=0, column=0, sticky="w")
            self._text_refs.append((lab, key, "text", {}))
            val = ttk.Label(cell, textvariable=var, font=self.fonts["smallb"])
            val.grid(row=1, column=0, sticky="w")
        for col in range(3):
            overview.columnconfigure(col, weight=1, uniform="overview")
        self._reg(ttk.Label(overview, text=T("lbl_two_losses"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=420, justify="left"),
                  "lbl_two_losses", font=self.fonts["tiny"]).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.view = NetworkView(center, self, self.fonts)
        self.view.pack(side="top", fill="both", expand=True)

    # -- правая колонка: пять вкладок --------------------------------------
    def _build_right(self):
        holder = ttk.Frame(self, padding=(6, 4, 12, 4))
        holder.grid(row=1, column=2, sticky="nsew")
        self.nb = ttk.Notebook(holder)
        self.nb.pack(fill="both", expand=True)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self.refresh())

        self.tab_step = ttk.Frame(self.nb, padding=(8, 8))
        self.tab_graph = ttk.Frame(self.nb, padding=(8, 8))
        self.tab_pred = ttk.Frame(self.nb, padding=(8, 8))
        self.tab_weights = ttk.Frame(self.nb, padding=(8, 8))
        self.tab_help = ttk.Frame(self.nb, padding=(8, 8))
        for tab, key in ((self.tab_step, "tab_detailed"), (self.tab_graph, "tab_graph"),
                         (self.tab_pred, "tab_pred"), (self.tab_weights, "tab_weights"),
                         (self.tab_help, "tab_help")):
            self.nb.add(tab, text=T(key))
            self._tabs.append((tab, key))

        # 1. Пошаговый режим
        top = ttk.Frame(self.tab_step)
        top.pack(side="top", fill="x")
        self.var_step_head = tk.StringVar()
        ttk.Label(top, textvariable=self.var_step_head, font=self.fonts["midb"]).pack(side="left")
        dots = ttk.Frame(self.tab_step)
        dots.pack(side="top", fill="x", pady=(2, 6))
        self.step_dots = []
        for n in range(1, TOTAL_STEPS + 1):
            dot = tk.Label(dots, text=str(n), width=2, relief="solid", bd=1, cursor="hand2",
                           font=self.fonts["smallb"], background="#eceae4", foreground="#6a6a63")
            dot.pack(side="left", padx=1)
            dot.bind("<Button-1>", lambda _e, k=n: self.goto_step(k))
            dot.bind("<Enter>", lambda _e, k=n: self._dot_hover(k))
            self.step_dots.append(dot)
        self.step_text = self._text_panel(self.tab_step)
        self._reg(ttk.Label(self.tab_step, text=T("det_mode_note"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=400, justify="left"),
                  "det_mode_note", font=self.fonts["tiny"]).pack(side="bottom", fill="x", pady=(4, 0))
        btns = ttk.Frame(self.tab_step)
        btns.pack(side="bottom", fill="x", pady=(6, 0))
        self.btn_prev = ttk.Button(btns, text=T("btn_prev"), command=self.prev_step)
        self.btn_prev.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.btn_next = ttk.Button(btns, text=T("btn_next"), command=self.next_step)
        self.btn_next.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        self.btn_full = ttk.Button(btns, text=T("btn_full"), command=self.show_full)
        self.btn_full.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.btn_step_reset = ttk.Button(btns, text=T("btn_reset_steps"), command=self.reset_steps)
        self.btn_step_reset.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        for b, key in ((self.btn_prev, "btn_prev"), (self.btn_next, "btn_next"),
                       (self.btn_full, "btn_full"), (self.btn_step_reset, "btn_reset_steps")):
            self._text_refs.append((b, key, "text", {}))

        # 2. График потерь
        self.graph = LossGraph(self.tab_graph, self, self.fonts)
        self.graph.pack(side="top", fill="both", expand=True)
        self._reg(ttk.Label(self.tab_graph, text=T("graph_caption"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=400, justify="left"),
                  "graph_caption", font=self.fonts["tiny"]).pack(side="bottom", fill="x", pady=(6, 0))

        # 3. Предсказания + сравнение до/после
        self._reg(ttk.Label(self.tab_pred, text=T("pred_caption"), font=self.fonts["small"],
                            wraplength=400, justify="left"),
                  "pred_caption", font=self.fonts["small"]).pack(anchor="w")
        pcols = ("n", "x1", "x2", "y", "pred", "err", "cmp")
        self.pred_table = ttk.Treeview(self.tab_pred, columns=pcols, show="headings",
                                       height=6, style="Pred.Treeview")
        # Ширины заданы по замерам: берём максимум из заголовка и самой длинной
        # ячейки плюс 5px с каждой стороны. Сумма держится ниже панели при 1300px
        # окна, поэтому ни одна колонка не ужимается растянением последней.
        widths = {"n": 45, "x1": 17, "x2": 17, "y": 42, "pred": 88, "err": 63, "cmp": 135}
        for col in pcols:
            self.pred_table.heading(col, text=self._pred_heading(col))
            self.pred_table.column(col, width=widths[col], minwidth=widths[col] - 8,
                                   anchor="center" if col != "cmp" else "w")
        self.pred_table.pack(fill="x", pady=(6, 4))
        self.pred_table.tag_configure("ok", foreground=COL_POS)
        self.pred_table.tag_configure("bad", foreground=COL_NEG)
        self.pred_table.tag_configure("current", background=COL_ACTIVE_FILL)
        self._reg(ttk.Label(self.tab_pred, text=T("cmp_title"),
                            font=self.fonts["smallb"]), "cmp_title",
                  font=self.fonts["smallb"]).pack(anchor="w", pady=(6, 0))
        self.var_cmp_before = tk.StringVar()
        self.var_cmp_now = tk.StringVar()
        self.var_cmp_change = tk.StringVar()
        for var, key in ((self.var_cmp_before, "cmp_avg_before"), (self.var_cmp_now, "cmp_avg_current"),
                         (self.var_cmp_change, "cmp_change")):
            lab = ttk.Label(self.tab_pred, textvariable=var, font=self.fonts["small"])
            lab.pack(anchor="w")
            self._text_refs.append((lab, key, "text", {}))
        self._reg(ttk.Label(self.tab_pred, text=T("cmp_note"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=400, justify="left"),
                  "cmp_note", font=self.fonts["tiny"]).pack(anchor="w", pady=(4, 0))
        self._reg(ttk.Label(self.tab_pred, text=T("pred_legend"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=400, justify="left"),
                  "pred_legend", font=self.fonts["tiny"]).pack(anchor="w")

        # 4. Веса
        self._reg(ttk.Label(self.tab_weights, text=T("box_weights"), font=self.fonts["small"]),
                  "box_weights", font=self.fonts["small"]).pack(anchor="w", pady=(0, 6))
        self.weight_fields = {}
        self.weight_init = {}
        groups = (("grp_h1", HIDDEN1_KEYS), ("grp_h2", HIDDEN2_KEYS), ("grp_o", OUTPUT_KEYS))
        for gkey, keys in groups:
            box = ttk.LabelFrame(self.tab_weights, text=T(gkey), padding=(8, 4))
            box.pack(fill="x", pady=(0, 6))
            self._text_refs.append((box, gkey, "text", {}))
            for key in keys:
                row = ttk.Frame(box)
                row.pack(fill="x", pady=1)
                field = ParamField(row, key, self.trainer.params[key], *LIMITS["w"],
                                   on_change=self.on_weight, fonts=self.fonts,
                                   symbol_key=key, hint_key="hint_" + key, label_width=4)
                field.pack(side="left")
                self.weight_fields[key] = field
                init = ttk.Label(row, text="", font=self.fonts["tiny"], foreground="#5a5a53")
                init.pack(side="right", padx=(6, 0))
                self.weight_init[key] = init
        wbtns = ttk.Frame(self.tab_weights)
        wbtns.pack(fill="x", pady=(4, 2))
        self.btn_random = ttk.Button(wbtns, text=T("btn_random"), command=self.randomize)
        self.btn_random.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_standard = ttk.Button(wbtns, text=T("btn_standard"), command=self.standard)
        self.btn_standard.pack(side="left", expand=True, fill="x")
        self._text_refs.append((self.btn_random, "btn_random", "text", {}))
        self._text_refs.append((self.btn_standard, "btn_standard", "text", {}))
        self._reg(ttk.Label(self.tab_weights, text=T("weights_note"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=400, justify="left"),
                  "weights_note", font=self.fonts["tiny"]).pack(anchor="w", pady=(6, 0))

        # 5. Справка
        self._reg(ttk.Label(self.tab_help, text=T("help_intro"), font=self.fonts["tiny"],
                            foreground="#5a5a53", wraplength=400, justify="left"),
                  "help_intro", font=self.fonts["tiny"]).pack(anchor="w", pady=(0, 6))
        # Список разделов слева съел бы половину ширины текста: selector — компактный
        # выпадающий список над текстовой панелью, текст занимает всю ширину.
        self.help_pick = ttk.Combobox(self.tab_help, state="readonly",
                                      font=self.fonts["small"], width=40)
        self.help_pick.pack(anchor="w", pady=(0, 4))
        self.help_pick.bind("<<ComboboxSelected>>", self.on_help_select)
        self.help_text = self._text_panel(self.tab_help)
        self._fill_help()

    # -- нижняя строка -----------------------------------------------------
    def _build_bottom(self):
        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 8))
        # Имя автора постоянно, внизу справа, не переводится и не является ключом локализации.
        # Упаковываем первым: иначе длинная строка статуса не оставит ему места.
        ttk.Label(bar, text="SteveWorks001", font=self.fonts["smallb"],
                  foreground="#8a8a82").pack(side="right")
        self.var_status = tk.StringVar()
        ttk.Label(bar, textvariable=self.var_status, font=self.fonts["small"]).pack(
            side="left", anchor="w")

    # -- вспомогательное ---------------------------------------------------
    def _text_panel(self, parent):
        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=True)
        text = tk.Text(holder, wrap="word", width=1, height=1, bg=COL_PAPER, fg=COL_TEXT,
                       highlightthickness=1, highlightbackground=COL_FRAME, bd=0, padx=10, pady=8,
                       font=self.fonts["mono"], spacing3=3, cursor="arrow", state="normal")
        bar = ttk.Scrollbar(holder, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return text

    @staticmethod
    def _set_text(widget, content, keep_view=False):
        top = widget.yview()[0] if keep_view else "1.0"
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")
        widget.yview_moveto(top) if keep_view else widget.see("1.0")

    def set_status(self, key, **kw):
        self.var_status.set(T(key, **kw))

    def _on_close(self):
        self.stop_training(silent=True)
        self.destroy()

    def step_focus(self):
        """Что подсветить на схеме: текущий шаг или «ничего» (вкладка не активна)."""
        if self.nb.nametowidget(self.nb.select()) is not self.tab_step:
            return "fwd", set(), set()
        phase, nodes, edges = STEP_FOCUS[self.step_no]
        return phase, set(nodes), set(edges)

    # -- язык --------------------------------------------------------------
    def switch_language(self, code):
        if code == LANG:
            return
        set_language(code)
        self.retranslate()
        self.set_status("status_lang")

    def retranslate(self):
        """Перевод без сброса: датасет, веса, эпоха, история и ввод остаются."""
        self.title(T("app_title"))
        for widget, key, attr, kw in self._text_refs:
            try:
                widget.configure(**{attr: T(key, **kw)})
            except tk.TclError:
                pass
        for tab, key in self._tabs:
            self.nb.tab(tab, text=T(key))
        for widget, key, code in self._radios:
            widget.configure(text=T(key))
        self.field_epochs.symbol.configure(text=T("lbl_epochs"))
        for key, field in self.weight_fields.items():
            field.retranslate()
        for col in self.ds_table["columns"]:
            self.ds_table.heading(col, text=self._ds_heading(col))
        for col in self.pred_table["columns"]:
            self.pred_table.heading(col, text=self._pred_heading(col))
        self._fill_help()
        self.refresh()

    @staticmethod
    def _ds_heading(col):
        return T("th_" + {"n": "sample", "y": "target"}.get(col, col))

    @staticmethod
    def _pred_heading(col):
        head = {"n": "sample", "y": "target", "pred": "prediction", "err": "error",
                "cmp": "compare"}.get(col, col)
        key = "th_" + head
        return T(key) if key in TEXTS[LANG] else col

    # -- обработчики -------------------------------------------------------
    def on_dataset(self):
        self.stop_training(silent=True)
        self.trainer.set_dataset(self.var_dataset.get())
        self.step_no = 1
        self.full_mode = False
        self.refresh()
        self.set_status("status_dataset", name=T("ds_" + self.trainer.dataset_id))

    def on_sample_select(self, _event):
        if self._syncing_table:
            return
        selection = self.ds_table.selection()
        if not selection:
            return
        index = int(selection[0])
        if index == self.trainer.sample_index:
            return
        self.trainer.sample_index = index
        self.step_no = 1
        self.full_mode = False
        self.refresh()
        self.set_status("status_sample", i=index + 1, n=self.trainer.n_samples)

    def on_epochs(self, value, raw, key, clamped=False):
        if value is None:
            self.set_status("status_bad", raw=raw)
            return
        if clamped:
            self.set_status("status_range", name=T("lbl_epochs"),
                            lo=LIMITS["epochs"][0], hi=LIMITS["epochs"][1])
        self.trainer.total_epochs = int(value)
        self.refresh()

    def on_eta(self, value, raw, key, clamped=False):
        if value is None:
            self.set_status("status_bad", raw=raw)
            return
        if clamped:
            self.set_status("status_range", name="η", lo=LIMITS["eta"][0], hi=LIMITS["eta"][1])
        self.trainer.eta = float(value)
        self.refresh()

    def on_weight(self, value, raw, key, clamped=False):
        if value is None:
            self.set_status("status_bad", raw=raw)
            return
        if clamped:
            self.set_status("status_range", name=key, lo=LIMITS["w"][0], hi=LIMITS["w"][1])
        # Ручная правка = новая отправная точка: история прошлой тренировки больше не описывает сеть.
        params = dict(self.trainer.params)
        params[key] = float(value)
        self.trainer.set_initial(params)
        self.step_no = 1
        self.full_mode = False
        self.refresh(skip_field=key)
        self.set_status("status_weight_set", name=key, value=fmt(value))

    # -- обучение ----------------------------------------------------------
    def step_epoch(self):
        if self._after_id is not None:
            self.set_status("status_stop_first")
            return
        if self.trainer.epoch >= self.trainer.total_epochs:
            self.set_status("status_completed", total=self.trainer.total_epochs)
            return
        avg, _losses = self.trainer.run_epoch()
        self.trainer.status = ("completed" if self.trainer.epoch >= self.trainer.total_epochs
                               else "ready")
        self.step_no = 1
        self.full_mode = False
        self.refresh()
        self.set_status("status_epoch_done", e=self.trainer.epoch, loss=fmt(avg))
        if self.trainer.status == "completed":
            self.set_status("status_completed", total=self.trainer.total_epochs)

    def start_training(self):
        if self._after_id is not None:
            return
        if self.trainer.epoch >= self.trainer.total_epochs:
            self.set_status("status_completed", total=self.trainer.total_epochs)
            return
        self.trainer.status = "training"
        self.set_status("status_training")
        self._schedule(0)
        self._set_training_buttons(True)

    def _schedule(self, delay):
        self._after_id = self.after(delay, self._tick)

    def _tick(self):
        self._after_id = None
        if self.trainer.status != "training":
            return
        per, delay = SPEEDS[self.var_speed.get()]
        for _ in range(per):
            if self.trainer.epoch >= self.trainer.total_epochs:
                self.trainer.status = "completed"
                self.refresh()
                self._set_training_buttons(False)
                self.set_status("status_completed", total=self.trainer.total_epochs)
                return
            self.trainer.run_epoch()
        self.refresh()
        self._schedule(delay)

    def stop_training(self, silent=False, _event=None):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self.trainer.status == "training":
            self.trainer.status = "stopped"
            self._set_training_buttons(False)
            if not silent:
                self.set_status("status_stopped", e=self.trainer.epoch)
        elif silent:
            self._set_training_buttons(False)

    def reset(self):
        self.stop_training(silent=True)
        self.trainer.reset_state()
        self.step_no = 1
        self.full_mode = False
        self.refresh()
        self.set_status("status_reset")

    def reset_history(self):
        if not self.trainer.history:
            self.set_status("status_no_history")
            return
        self.trainer.clear_history()
        self.refresh()
        self.set_status("status_history_cleared")

    def randomize(self):
        self.stop_training(silent=True)
        self.trainer.set_initial(random_params())
        self.step_no = 1
        self.full_mode = False
        self.refresh()
        self.set_status("status_random")

    def standard(self):
        self.stop_training(silent=True)
        self.trainer.set_initial(dict(STANDARD_PARAMS))
        self.step_no = 1
        self.full_mode = False
        self.refresh()
        self.set_status("status_standard")

    def _set_training_buttons(self, running):
        self.btn_train.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")
        self.btn_step.configure(state="disabled" if running else "normal")

    # -- пошаговый режим ---------------------------------------------------
    def goto_step(self, n):
        self.full_mode = False
        self.step_no = clamp(n, 1, TOTAL_STEPS)
        self.refresh()
        self.set_status("status_step", n=self.step_no, total=TOTAL_STEPS,
                        title=T("stp%d" % self.step_no))

    def next_step(self):
        if self.step_no >= TOTAL_STEPS:
            self.set_status("status_last_step")
            return
        self.goto_step(self.step_no + 1)

    def prev_step(self):
        if self.step_no <= 1:
            self.set_status("status_first_step")
            return
        self.goto_step(self.step_no - 1)

    def reset_steps(self):
        self.goto_step(1)
        self.set_status("status_steps_reset")

    def show_full(self):
        self.full_mode = True
        self.refresh()
        self.set_status("status_full_mode", total=TOTAL_STEPS,
                        i=self.trainer.sample_index + 1)

    def _dot_hover(self, n):
        self.var_step_head.set(T("stp%d" % n))

    # -- справка -----------------------------------------------------------
    def _fill_help(self):
        titles = T("help_titles")
        self.help_pick["values"] = ["%d. %s" % (i, title)
                                    for i, title in enumerate(titles, 1)]
        self.help_pick.current(self.help_index)
        self._show_help(self.help_index)

    def on_help_select(self, _event):
        self.help_index = max(self.help_pick.current(), 0)
        self._show_help(self.help_index)

    def _show_help(self, index):
        titles = T("help_titles")
        body = T("help_b%d" % (index + 1))
        self._set_text(self.help_text, "%s\n\n%s" % (titles[index], body))

    # -- отрисовка состояний -----------------------------------------------
    def step_context(self):
        return {
            "n_samples": self.trainer.n_samples,
            "rows": self.trainer.rows,
            "dataset_name": T("ds_" + self.trainer.dataset_id),
            "sample_no": self.trainer.sample_index + 1,
            "next_sample": ((self.trainer.sample_index + 1) % self.trainer.n_samples) + 1,
        }

    def refresh(self, skip_field=None):
        trainer = self.trainer
        self.var_epoch.set("%d / %d" % (trainer.epoch, trainer.total_epochs))
        self.var_eta_show.set(fmt(trainer.eta))
        self.var_sample_of.set("%d / %d" % (trainer.sample_index + 1, trainer.n_samples))
        x1, x2, y = trainer.current_sample
        self.var_x1.set(T("lbl_x1", v=inp(x1)))
        self.var_x2.set(T("lbl_x2", v=inp(x2)))
        self.var_y.set(T("lbl_target", v=inp(y)))
        self.var_epoch_avg.set(fmt(trainer.epoch_avg_loss) if trainer.epoch_avg_loss is not None
                               else "—")
        self.var_eval.set(fmt(trainer.eval_loss))
        # Потеря текущего примера: если пример ещё не проходил через обучение,
        # считаем её на текущих весах — число всё равно настоящее.
        if trainer.last_step and trainer.last_step["index"] == trainer.sample_index:
            self.var_sample_loss.set(fmt(trainer.last_sample_loss))
        else:
            self.var_sample_loss.set(fmt(sample_loss(forward(x1, x2, trainer.params)["y_hat"], y)))
        self.var_correct.set("%d / %d" % (trainer.best_prediction_count, trainer.n_samples))
        self.var_before.set(fmt(trainer.reset_eval))
        self.var_now.set(fmt(trainer.eval_loss))
        self.var_status_line.set(T(STATE_KEY.get(trainer.status, "state_ready")))

        self._refresh_dataset_table()
        self._refresh_predictions()
        self._refresh_step_text()
        self._refresh_weights(skip_field)
        self.view.redraw()
        self.graph.redraw()

    def _refresh_dataset_table(self):
        rows = self.trainer.rows
        for i, (x1, x2, y) in enumerate(rows):
            values = (i + 1, inp(x1), inp(x2), inp(y))
            if self.ds_table.exists(str(i)):
                self.ds_table.item(str(i), values=values)
            else:
                self.ds_table.insert("", "end", iid=str(i), values=values)
        count = len(rows)
        for stale in [i for i in self.ds_table.get_children() if int(i) >= count]:
            self.ds_table.delete(stale)
        current = str(self.trainer.sample_index)
        for iid in self.ds_table.get_children():
            self.ds_table.item(iid, tags=("current",) if iid == current else ())
        # Программный выбор строки порождает <<TreeviewSelect>>: без флага
        # обработчик вызвал бы refresh() рекурсивно.
        self._syncing_table = True
        self.ds_table.selection_set(current)
        self._syncing_table = False

    def _refresh_predictions(self):
        preds = self.trainer.predictions()
        initial = self.trainer.initial
        for i, row in enumerate(preds):
            before = forward(row["x1"], row["x2"], initial)["y_hat"]
            values = (i + 1, inp(row["x1"]), inp(row["x2"]), inp(row["y"]),
                      fmt(row["y_hat"]), signed(row["error"]),
                      "%s → %s → %s" % (fmt(before, 3), fmt(row["y_hat"], 3), inp(row["y"])))
            tags = []
            tags.append("ok" if (row["y_hat"] >= 0.5) == (row["y"] >= 0.5) else "bad")
            if i == self.trainer.sample_index:
                tags.append("current")
            if self.pred_table.exists(str(i)):
                self.pred_table.item(str(i), values=values, tags=tags)
            else:
                self.pred_table.insert("", "end", iid=str(i), values=values, tags=tags)
        for stale in [i for i in self.pred_table.get_children() if int(i) >= len(preds)]:
            self.pred_table.delete(stale)
        self.var_cmp_before.set("%s: %s" % (T("cmp_avg_before"), fmt(self.trainer.reset_eval)))
        self.var_cmp_now.set("%s: %s" % (T("cmp_avg_current"), fmt(self.trainer.eval_loss)))
        delta = self.trainer.eval_loss - self.trainer.reset_eval
        # До и сейчас уже показаны двумя строками выше: здесь только приращение.
        self.var_cmp_change.set("%s: %s" % (T("cmp_change"), signed(delta)))

    def _refresh_step_text(self):
        if self.nb.nametowidget(self.nb.select()) is not self.tab_step:
            return
        snapshot = snapshot_step(self.trainer)
        ctx = self.step_context()
        if self.full_mode:
            self._set_text(self.step_text, render_all(snapshot, ctx), keep_view=True)
            self.var_step_head.set(T("det_full", i=ctx["sample_no"]))
        else:
            self._set_text(self.step_text, render_step(self.step_no, snapshot, ctx))
            self.var_step_head.set(step_head(self.step_no))
        for n, dot in enumerate(self.step_dots, 1):
            if self.full_mode:
                bg, fg = "#e6efe7", COL_POS
            elif n == self.step_no:
                bg, fg = COL_ACTIVE, "#ffffff"
            elif n < self.step_no:
                bg, fg = "#e6efe7", COL_POS
            else:
                bg, fg = "#eceae4", "#6a6a63"
            dot.configure(background=bg, foreground=fg)

    def _refresh_weights(self, skip_field=None):
        focused = self.focus_get()
        for key, field in self.weight_fields.items():
            # Подпись «начало» описывает initial, а не поле ввода, поэтому
            # обновляется всегда — иначе она зависала бы после смены языка.
            self.weight_init[key].configure(
                text="%s: %s" % (T("lbl_init"), fmt(self.trainer.initial[key], 3)))
            if key == skip_field or (focused is not None and focused is field.entry):
                continue
            field.set_value(self.trainer.params[key])


# ---------------------------------------------------------------
# 11. Точка входа
# ---------------------------------------------------------------

def main():
    app = TrainingLabApp()
    app.mainloop()


if __name__ == "__main__":
    main()
