# INVESTOR — Полное Техническое Задание
## Система автоматической торговли на основе волновой теории Эллиотта (БКС Trade API)

**Версия:** 2.0 (финальная, исправленная)
**Дата:** 21.03.2026
**Заказчик:** Клевакин Алексей
**Статус:** Действующий документ — ВСЕ решения принимаются на основе этого файла

---

## Содержание

1. [Цель и описание проекта](#1-цель-и-описание-проекта)
2. [Исправления первичного ТЗ](#2-исправления-первичного-тз-критически-важно)
3. [Архитектура системы](#3-архитектура-системы)
4. [Схема базы данных](#4-схема-базы-данных)
5. [Этап 1: Каркас проекта](#5-этап-1-каркас-проекта-и-конфигурация)
6. [Этап 2: Волновой анализ и бэктестинг](#6-этап-2-волновой-анализ-и-бэктестинг)
7. [Этап 3: БКС API и управление данными](#7-этап-3-бкс-api-и-управление-данными)
8. [Этап 4: Генерация сигналов и риск-менеджмент](#8-этап-4-генерация-сигналов-и-риск-менеджмент)
9. [Этап 5: Исполнение сделок](#9-этап-5-исполнение-сделок-и-стоп-лосс)
10. [Этап 6: Отчётность и уведомления](#10-этап-6-отчётность-и-уведомления)
11. [Этап 7: Веб-интерфейс](#11-этап-7-веб-интерфейс)
12. [Безопасность](#12-требования-к-безопасности)
13. [Тестирование](#13-стратегия-тестирования)
14. [Глоссарий](#14-глоссарий)

---

## 1. Цель и описание проекта

**INVESTOR** — десктопное веб-приложение (FastAPI-сервер + браузер-интерфейс) для автоматической торговли акциями на Московской бирже (ММВБ/MOEX) через брокера БКС.

### Что делает система:
1. Подключается к **БКС Trade API** и загружает исторические + потоковые свечные данные.
2. Находит **5-волновые импульсные структуры** (волновая теория Эллиотта) на исторических данных.
3. Генерирует **торговые сигналы** (покупка/продажа) по строгим правилам.
4. Рассчитывает **размер позиции** с учётом риска и лотности ММВБ.
5. **Выставляет заявки** через API и отслеживает стоп-лоссы программно.
6. Ведёт **журнал** всех сделок и формирует финансовые отчёты.
7. Отправляет **уведомления** в мессенджер MAX (VK).
8. Отображает всё через **веб-интерфейс** с интерактивными графиками.

### Порядок разработки (принятое решение):
> **СНАЧАЛА бэктестинг** — стратегия проверяется на истории ДО подключения к реальному брокеру.
> Это защищает от потери денег из-за ошибок в коде.

---

## 2. Исправления первичного ТЗ (критически важно)

### 2.1. Ошибка в количестве свинг-точек

**БЫЛО (неверно):** «ищем последовательность из 10 свинг-точек»
**СТАЛО (верно):** 5-волновой импульс = **6 опорных точек**

```
Восходящий импульс:
  low₀ ──► high₁ ──► low₂ ──► high₃ ──► low₄ ──► high₅
  |         |         |         |         |         |
  Начало  Конец W1  Конец W2  Конец W3  Конец W4  Конец W5

Нисходящий импульс — зеркально:
  high₀ ──► low₁ ──► high₂ ──► low₃ ──► high₄ ──► low₅
```

Именование в коде:
- `p0` — начало импульса
- `p1` — конец волны 1 (для восходящего: локальный max)
- `p2` — конец волны 2 (для восходящего: локальный min)
- `p3` — конец волны 3 (для восходящего: локальный max)
- `p4` — конец волны 4 (для восходящего: локальный min)
- `p5` — конец волны 5 (для восходящего: локальный max)

### 2.2. Архитектура: async-first

Все I/O операции (HTTP, WebSocket, SQLite) выполняются в `asyncio`.
Тяжёлые вычисления (волновой анализ) → `ThreadPoolExecutor`.
Веб-сервер: **FastAPI** (нативно async), не Flask.

### 2.3. Стоп-заявки в БКС API

БКС Trade API поддерживает только **лимитные и рыночные заявки** (данные на 03.2026).
Стоп-лосс реализуется **программно**: WebSocket-подписка на тики → при достижении уровня → рыночная заявка.
**При потере WebSocket:** немедленное закрытие ВСЕХ открытых позиций по рынку + уведомление в MAX.

### 2.4. Инструменты

Система торгует несколькими инструментами одновременно. Инструменты задаются в `config.py`.
Стартовый список (настраивается): `SBER`, `GAZP`, `LKOH`, `YNDX`, `GMKN`.
У каждого инструмента разный `lot_size` — это **обязательно учитывается** при расчёте позиции.

---

## 3. Архитектура системы

### 3.1. Структура директорий

```
c:\investor\
│
├── .env                        # Секреты: токены (НЕ в git)
├── .env.example                # Шаблон .env
├── .gitignore
├── requirements.txt
├── README.md
├── full_tz.md                  # Этот файл (главный документ)
├── constants.py                # Справочник проекта для AI
├── config.py                   # Параметры стратегии (без секретов)
├── main.py                     # Точка входа / оркестратор
│
├── broker/
│   ├── __init__.py
│   ├── base.py                 # AbstractBrokerClient (интерфейс)
│   └── bcs_client.py           # Реализация через BCSPy / httpx
│
├── data/
│   ├── __init__.py
│   ├── manager.py              # DataManager: загрузка, кэш, ATR
│   ├── db.py                   # SQLite: создание таблиц, CRUD
│   └── investor.db             # SQLite база данных (создаётся автоматически)
│
├── analysis/
│   ├── __init__.py
│   ├── indicators.py           # ATR, RSI (numpy/pandas)
│   ├── swing_detector.py       # Поиск свинг-точек
│   └── wave_analyzer.py        # Идентификация 5-волновых импульсов
│
├── signals/
│   ├── __init__.py
│   └── generator.py            # SignalGenerator → Signal
│
├── risk/
│   ├── __init__.py
│   └── manager.py              # RiskManager + circuit breaker
│
├── execution/
│   ├── __init__.py
│   └── executor.py             # OrderExecutor, программный стоп-лосс
│
├── reporting/
│   ├── __init__.py
│   └── reporter.py             # Метрики, equity curve, Excel
│
├── notifications/
│   ├── __init__.py
│   └── max_bot.py              # Уведомления в MAX (maxapi)
│
├── backtesting/
│   ├── __init__.py
│   └── engine.py               # Бэктест-движок
│
├── web/
│   ├── __init__.py
│   ├── app.py                  # FastAPI приложение
│   ├── routers/
│   │   ├── dashboard.py
│   │   ├── charts.py
│   │   ├── trades.py
│   │   └── settings.py
│   ├── static/
│   │   ├── css/main.css
│   │   └── js/main.js
│   └── templates/
│       ├── base.html
│       ├── dashboard.html
│       ├── chart.html
│       └── trades.html
│
├── logs/                       # Создаётся автоматически
│   └── investor_YYYY-MM-DD.log
│
└── tests/
    ├── __init__.py
    ├── test_swing_detector.py
    ├── test_wave_analyzer.py
    ├── test_risk_manager.py
    └── test_backtest_engine.py
```

### 3.2. Взаимодействие модулей

```
                    ┌─────────────┐
                    │   main.py   │ ← Оркестратор, event loop
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌─────────────┐ ┌───────────┐ ┌────────────┐
    │ BrokerClient│ │DataManager│ │  Reporter  │
    │  (broker/)  │ │  (data/)  │ │(reporting/)│
    └──────┬──────┘ └─────┬─────┘ └────────────┘
           │              │
           └──────┬───────┘
                  ▼
          ┌───────────────┐
          │ WaveAnalyzer  │
          │  (analysis/)  │
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │SignalGenerator│
          │  (signals/)   │
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │  RiskManager  │◄── capital от BrokerClient
          │   (risk/)     │
          └───────┬───────┘
                  ▼
          ┌───────────────┐     ┌──────────────┐
          │ OrderExecutor │────►│  MAX Notifier│
          │ (execution/)  │     │(notificatio/)│
          └───────────────┘     └──────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  web/app.py   │ ← FastAPI: дашборд, графики
          └───────────────┘
```

### 3.3. Машина состояний сделки

```
IDLE
  │ signal detected
  ▼
SIGNAL_PENDING
  │ risk check passed
  ▼
ORDER_PLACED
  │ order filled
  ▼
POSITION_OPEN ──── stop_loss hit ──► CLOSING ──► CLOSED
  │                                                  ▲
  │ take_profit hit ────────────────────────────────┤
  │ manual close ───────────────────────────────────┤
  │ opposite signal ────────────────────────────────┘
  │ circuit breaker ────────────────────────────────┘
```

---

## 4. Схема базы данных

База: `data/investor.db` (SQLite). При необходимости мигрируем на PostgreSQL без изменения кода (через SQLAlchemy).

### Таблица `candles`

```sql
CREATE TABLE candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,        -- '1H', '1D', '5M' и т.д.
    dt          TEXT    NOT NULL,        -- ISO 8601: '2024-01-15T10:00:00'
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    atr14       REAL,                    -- рассчитывается при загрузке
    UNIQUE(symbol, interval, dt)
);
CREATE INDEX idx_candles_symbol_interval ON candles(symbol, interval, dt);
```

### Таблица `signals`

```sql
CREATE TABLE signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    direction       TEXT    NOT NULL,   -- 'BUY' / 'SELL'
    entry_price     REAL    NOT NULL,
    stop_loss       REAL    NOT NULL,
    take_profit     REAL    NOT NULL,
    wave_json       TEXT,               -- JSON-снимок структуры волн
    reason          TEXT,               -- текстовое описание сигнала
    acted_on        INTEGER DEFAULT 0,  -- 1 если сигнал исполнен
    backtest_run    INTEGER DEFAULT 0   -- 1 если сигнал из бэктеста
);
```

### Таблица `trades`

```sql
CREATE TABLE trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER REFERENCES signals(id),
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,   -- 'BUY' / 'SELL'
    lot_size        INTEGER NOT NULL,   -- размер лота брокера
    quantity        INTEGER NOT NULL,   -- количество лотов
    entry_price     REAL    NOT NULL,
    entry_time      TEXT    NOT NULL,
    exit_price      REAL,
    exit_time       TEXT,
    commission      REAL    DEFAULT 0,
    pnl             REAL,               -- прибыль/убыток в рублях
    pnl_pct         REAL,               -- прибыль/убыток в %
    exit_reason     TEXT,               -- 'stop_loss'/'take_profit'/'manual'/'signal'
    status          TEXT    DEFAULT 'OPEN' -- 'OPEN'/'CLOSED'
);
```

### Таблица `app_logs`

```sql
CREATE TABLE app_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    level       TEXT    NOT NULL,   -- 'INFO'/'WARNING'/'ERROR'/'CRITICAL'
    module      TEXT    NOT NULL,
    message     TEXT    NOT NULL
);
```

### Таблица `instruments`

```sql
CREATE TABLE instruments (
    symbol      TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,       -- 'Сбербанк'
    lot_size    INTEGER NOT NULL,       -- SBER=10, GAZP=10, YNDX=1
    min_price_step REAL NOT NULL,       -- минимальный шаг цены
    currency    TEXT    DEFAULT 'RUB',
    is_active   INTEGER DEFAULT 1
);
```

---

## 5. Этап 1: Каркас проекта и конфигурация

**Цель:** создать структуру папок, зависимости, конфиг, логирование. После этапа: проект запускается без ошибок.

### 5.1. Файлы зависимостей

**`requirements.txt`:**
```
# Data processing
pandas==2.2.3
numpy==1.26.4

# Async HTTP / WebSocket
aiohttp==3.10.10
httpx==0.27.2

# Broker API
BCSPy>=0.1.0           # https://github.com/cia76/BCSPy

# Database
SQLAlchemy==2.0.36
aiosqlite==0.20.0

# Web interface
fastapi==0.115.5
uvicorn[standard]==0.32.1
jinja2==3.1.4
python-multipart==0.0.12

# Charts
plotly==5.24.1
openpyxl==3.1.5

# Config / Security
python-dotenv==1.0.1
cryptography==43.0.3

# Notifications (MAX messenger)
maxapi==0.9.17         # https://pypi.org/project/maxapi/

# Testing
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-cov==5.0.0
```

### 5.2. Переменные окружения

**`.env.example`** (шаблон — копировать в `.env` и заполнить):
```env
# БКС Trade API
BCS_CLIENT_ID=ваш_client_id
BCS_CLIENT_SECRET=ваш_client_secret
BCS_ACCOUNT_ID=ваш_номер_счёта

# MAX Messenger Bot
MAX_BOT_TOKEN=ваш_токен_бота
MAX_USER_ID=ваш_id_в_max

# Приложение
APP_HOST=127.0.0.1
APP_PORT=8000
LOG_LEVEL=INFO
```

### 5.3. Конфигурация стратегии

**`config.py`** — все настраиваемые параметры стратегии (без секретов):
```python
# Список торгуемых инструментов
SYMBOLS = ['SBER', 'GAZP', 'LKOH', 'YNDX', 'GMKN']

# Таймфрейм анализа
INTERVAL = '1H'          # '5M', '15M', '1H', '4H', '1D'
HISTORY_DAYS = 120       # глубина истории для загрузки

# Параметры волнового анализа
ATR_PERIOD = 14          # период ATR
ATR_MULTIPLIER = 1.5     # мин. движение = ATR * ATR_MULTIPLIER
LOOKBACK = 5             # окно в свечах для локального экстремума
FIB_TOLERANCE = 0.10     # допуск Фибоначчи ±10%

# Уровни Фибоначчи для волн
WAVE2_RETRACE_MIN = 0.382
WAVE2_RETRACE_MAX = 0.618
WAVE3_EXT_MIN = 1.618
WAVE3_EXT_MAX = 2.618
WAVE4_RETRACE_MIN = 0.382
WAVE4_RETRACE_MAX = 0.618
WAVE5_EXT_MIN = 0.618
WAVE5_EXT_MAX = 1.618

# Параметры риска
RISK_PERCENT = 0.02      # риск на сделку = 2% капитала
MAX_POSITIONS = 3        # макс. открытых позиций одновременно
MAX_CAPITAL_PER_TRADE = 0.20  # не более 20% капитала в одной позиции
CIRCUIT_BREAKER_LOSSES = 3    # остановка после N убытков подряд
SLIPPAGE_TICKS = 2       # допустимое проскальзывание в тиках

# RSI (опциональный фильтр)
RSI_PERIOD = 14
RSI_DIVERGENCE_ENABLED = True

# Комиссия брокера (для бэктестинга)
COMMISSION_PERCENT = 0.0005   # 0.05% за сделку (БКС)

# Веб-сервер
APP_HOST = '127.0.0.1'
APP_PORT = 8000
```

### 5.4. Логирование

**Формат:** файл `logs/investor_YYYY-MM-DD.log` + вывод в консоль.
**Уровни:** DEBUG (dev) / INFO (prod) / WARNING / ERROR / CRITICAL.
**Реализация:** стандартный модуль `logging` с `RotatingFileHandler`.

Каждое сообщение лога также пишется в таблицу `app_logs` в SQLite (уровень ERROR и выше).

### 5.5. Ресурсы этапа 1

- Python `logging` документация: https://docs.python.org/3/library/logging.html
- `python-dotenv`: https://python-dotenv.readthedocs.io/en/latest/
- `asyncio` основы: https://docs.python.org/3/library/asyncio.html
- SQLite WAL mode (производительность): https://www.sqlite.org/wal.html
- FastAPI документация: https://fastapi.tiangolo.com/
- `aiosqlite` (async SQLite): https://aiosqlite.omnilib.dev/en/stable/

---

## 6. Этап 2: Волновой анализ и бэктестинг

**Цель:** реализовать алгоритм поиска волн Эллиотта и проверить стратегию на исторических данных.
После этапа: система находит волны на CSV-файлах и показывает метрики бэктеста.

### 6.1. Расчёт индикаторов

**Файл:** `analysis/indicators.py`

#### ATR (Average True Range)

```
TR[i] = max(
    high[i] - low[i],
    |high[i] - close[i-1]|,
    |low[i]  - close[i-1]|
)
ATR[i] = EMA(TR, period=14)
```

Реализация через `pandas` + `numpy`. Первые `period-1` строк → NaN.

#### RSI (Relative Strength Index)

```
delta = close.diff()
gain  = delta.clip(lower=0)
loss  = (-delta).clip(lower=0)
avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
RS  = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))
```

### 6.2. Поиск свинг-точек

**Файл:** `analysis/swing_detector.py`

**Алгоритм:**

```python
def find_swings(df: pd.DataFrame, lookback: int = 5,
                atr_multiplier: float = 1.5) -> list[SwingPoint]:
    """
    Вход: DataFrame с колонками [datetime, high, low, close, atr14]
    Выход: список SwingPoint(idx, price, type='high'|'low', dt)
    """
    swings = []
    min_move = df['atr14'] * atr_multiplier

    for i in range(lookback, len(df) - lookback):
        # Локальный максимум: high[i] > all highs в окне [i-lookback, i+lookback]
        if df['high'][i] == df['high'][i-lookback:i+lookback+1].max():
            # Фильтр: значимое движение от предыдущего свинга
            if can_add_swing(swings, df['high'][i], min_move[i], 'high'):
                swings.append(SwingPoint(idx=i, price=df['high'][i],
                                         type='high', dt=df['datetime'][i]))

        elif df['low'][i] == df['low'][i-lookback:i+lookback+1].min():
            if can_add_swing(swings, df['low'][i], min_move[i], 'low'):
                swings.append(SwingPoint(idx=i, price=df['low'][i],
                                          type='low', dt=df['datetime'][i]))

    return swings
```

**Правило чередования:** точка типа `high` не может следовать за `high` — только за `low`.
При конфликте берём точку с более экстремальным значением.

### 6.3. Идентификация волн Эллиотта

**Файл:** `analysis/wave_analyzer.py`

#### Структуры данных

```python
@dataclass
class ImpulseWave:
    direction: str          # 'UP' / 'DOWN'
    p0: SwingPoint          # начало
    p1: SwingPoint          # конец волны 1
    p2: SwingPoint          # конец волны 2
    p3: SwingPoint          # конец волны 3
    p4: SwingPoint          # конец волны 4
    p5: SwingPoint          # конец волны 5
    confidence: float       # 0.0–1.0 (сколько правил выполнено)
    fib_ratios: dict        # фактические соотношения Фибоначчи
```

#### Алгоритм поиска

```
find_impulse(swings):
  Для каждого i от 0 до len(swings)-6:
    Берём 6 последовательных свингов: s[i]..s[i+5]
    Проверяем: чередование типов (low,high,low,high,low,high для UP)
    Проверяем 5 правил Эллиотта (см. ниже)
    Если confidence > threshold (напр. 0.7): добавляем в результат
  Возвращаем список найденных структур (сортировка: последние первыми)
```

#### 5 правил Эллиотта + Фибоначчи для восходящего импульса

| Правило | Проверка | Вес |
|---------|----------|-----|
| R1: Волна 2 не ниже начала | `p2.price > p0.price` | Обязательное |
| R2: Волна 3 не самая короткая | `len(W3) > min(len(W1), len(W5))` | Обязательное |
| R3: Волна 4 не перекрывает волну 1 | `p4.price > p1.price` | Обязательное |
| R4: Волны 1,3,5 идут вверх | `p1>p0, p3>p2, p5>p4` | Обязательное |
| F1: Волна 2 = 0.382–0.618 от волны 1 | `(p1-p2)/(p1-p0) ∈ [0.382-tol, 0.618+tol]` | Весовое |
| F2: Волна 3 = 1.618–2.618 от волны 1 | `(p3-p2)/(p1-p0) ∈ [1.618-tol, 2.618+tol]` | Весовое |
| F3: Волна 4 = 0.382–0.618 от волны 3 | `(p3-p4)/(p3-p2) ∈ [0.382-tol, 0.618+tol]` | Весовое |
| F4: Волна 5 = 0.618–1.618 от волны 1 | `(p5-p4)/(p1-p0) ∈ [0.618-tol, 1.618+tol]` | Весовое |

`confidence = (выполненных весовых правил) / 4`
Структура принимается только если ВСЕ 4 обязательных правила выполнены.

### 6.4. Бэктестинг

**Файл:** `backtesting/engine.py`

**Источники данных для бэктеста:**
- CSV-файлы с MOEX (загрузка вручную: https://www.finam.ru/profile/moex-akcii/sberbank/export/)
- Формат: `datetime, open, high, low, close, volume`

**Алгоритм бэктеста:**

```
1. Загрузить CSV → DataFrame
2. Рассчитать ATR14, RSI14
3. Для каждой свечи i (от lookback до конца):
   a. Получить свечи [0..i] (как будто мы в моменте i)
   b. Найти свинги на [0..i]
   c. Найти завершённые импульсы
   d. Если есть новый сигнал И нет открытой позиции:
      - Рассчитать entry, sl, tp
      - Рассчитать размер позиции (с учётом lot_size)
      - Открыть виртуальную позицию (записать в список)
   e. Для каждой открытой позиции:
      - Проверить low[i] <= stop_loss → закрыть по stop_loss
      - Проверить high[i] >= take_profit → закрыть по take_profit
4. Собрать все закрытые сделки → рассчитать метрики
```

**Метрики:**
- Total Return (%)
- Max Drawdown (%)
- Sharpe Ratio = (avg_return - risk_free_rate) / std_return × sqrt(252)
- Win Rate (% прибыльных сделок)
- Avg Win / Avg Loss
- Profit Factor = gross_profit / gross_loss
- Total Trades

**Команда запуска бэктеста:**
```bash
python -m backtesting.engine --symbol SBER --interval 1H --file data/SBER_1H.csv
```

### 6.5. Визуализация волнового анализа

**Библиотека:** Plotly (интерактивные графики).

График включает:
- Свечной график (candlestick)
- Нанесённые свинг-точки (синие точки)
- Нанесённые волны (цветные линии: W1-W5)
- Уровни стоп-лосса и тейк-профита (горизонтальные линии)

**Пример создания Plotly-графика:** https://plotly.com/python/candlestick-charts/

### 6.6. Ресурсы этапа 2

- Волновая теория Эллиотта (академическое описание): https://www.investopedia.com/articles/technical/111401.asp
- ATR расчёт и применение: https://www.investopedia.com/terms/a/atr.asp
- Числа Фибоначчи в трейдинге: https://www.investopedia.com/articles/technical/04/033104.asp
- `pandas` документация: https://pandas.pydata.org/docs/
- `numpy` документация: https://numpy.org/doc/stable/
- Plotly Candlestick: https://plotly.com/python/candlestick-charts/
- Plotly Python: https://plotly.com/python/
- Финам экспорт данных (для бэктеста): https://www.finam.ru/profile/moex-akcii/sberbank/export/
- Sharpe Ratio формула: https://www.investopedia.com/terms/s/sharperatio.asp
- Backtrader (альтернативный движок для сравнения): https://www.backtrader.com/docu/

---

## 7. Этап 3: БКС API и управление данными

**Цель:** подключиться к БКС Trade API, загружать реальные данные, кэшировать в SQLite.

### 7.1. Абстрактный клиент брокера

**Файл:** `broker/base.py`

```python
from abc import ABC, abstractmethod
import pandas as pd

class AbstractBrokerClient(ABC):

    @abstractmethod
    async def get_candles(self, symbol: str, interval: str,
                          start: str, end: str) -> pd.DataFrame:
        """Возвращает DataFrame: [datetime, open, high, low, close, volume]"""

    @abstractmethod
    async def get_portfolio(self) -> dict:
        """Возвращает: {cash: float, positions: [{symbol, qty, avg_price}]}"""

    @abstractmethod
    async def get_instrument_info(self, symbol: str) -> dict:
        """Возвращает: {lot_size, min_price_step, name, ...}"""

    @abstractmethod
    async def place_order(self, symbol: str, side: str, quantity: int,
                          order_type: str, price: float = None) -> str:
        """Возвращает order_id. side: 'BUY'/'SELL'. order_type: 'MARKET'/'LIMIT'"""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Отменяет заявку. Возвращает True если успешно."""

    @abstractmethod
    async def subscribe_candles(self, symbol: str, interval: str, callback) -> None:
        """WebSocket: callback(candle: dict) вызывается при новой свече."""

    @abstractmethod
    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """WebSocket: callback(tick: dict) вызывается при изменении цены."""
```

### 7.2. Реализация BCSPy-клиента

**Файл:** `broker/bcs_client.py`

Использует библиотеку **BCSPy** (https://github.com/cia76/BCSPy).
Если BCSPy недоступна → реализовать напрямую через `aiohttp` (REST) и `websockets` (WS).

**OAuth2 авторизация:**
- Токен получается через `client_credentials` flow.
- Сохраняется в памяти (не в файл, не в код).
- Автоматически обновляется за 60 секунд до истечения.
- Хранятся только `BCS_CLIENT_ID` и `BCS_CLIENT_SECRET` в `.env`.

**WebSocket reconnect стратегия:**
```
При разрыве соединения:
  attempt = 1
  while attempt <= 10:
    wait = min(2^attempt, 60) секунд
    попытка переподключения
    if успех: break
    attempt += 1
  if attempt > 10:
    КРИТИЧЕСКАЯ ОШИБКА → закрыть все позиции → уведомить → остановиться
```

### 7.3. DataManager

**Файл:** `data/manager.py`

**Методы:**
```python
class DataManager:
    async def initialize(self, symbols: list, interval: str, days: int):
        """При старте: проверяет БД, догружает недостающие данные."""

    async def get_candles(self, symbol: str, from_dt: str,
                          to_dt: str) -> pd.DataFrame:
        """Возвращает свечи с ATR14 и RSI14 из кэша (SQLite)."""

    async def on_new_candle(self, symbol: str, candle: dict) -> None:
        """Callback: сохраняет новую свечу в БД, пересчитывает ATR."""

    async def get_instrument_info(self, symbol: str) -> dict:
        """Кэширует справочную информацию по инструменту."""
```

**Логика кэширования:**
```
get_candles(symbol, from, to):
  1. Запросить из SQLite: SELECT * WHERE symbol=? AND dt BETWEEN ? AND ?
  2. Найти пропуски (gaps) во временном ряду
  3. Для каждого gap: запросить у брокера через REST
  4. Сохранить новые свечи в SQLite
  5. Вернуть полный DataFrame
```

### 7.4. Ресурсы этапа 3

- BCSPy (Python-клиент для БКС): https://github.com/cia76/BCSPy
- Альтернативный Go-клиент для БКС (справочно): https://github.com/tigusigalpa/bcs-trade-go
- Smart-Lab о БКС Trade API: https://smart-lab.ru/company/bcs/blog/1228126.php
- OAuth2 Client Credentials Flow: https://oauth.net/2/grant-types/client-credentials/
- `aiohttp` WebSocket client: https://docs.aiohttp.org/en/stable/client_websocket.html
- `aiosqlite` документация: https://aiosqlite.omnilib.dev/en/stable/
- SQLAlchemy async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

---

## 8. Этап 4: Генерация сигналов и риск-менеджмент

### 8.1. Генератор сигналов

**Файл:** `signals/generator.py`

#### Структура сигнала

```python
@dataclass
class Signal:
    symbol: str
    direction: str          # 'BUY' / 'SELL'
    entry_price: float      # цена входа
    stop_loss: float        # уровень стоп-лосса
    take_profit: float      # уровень тейк-профита
    wave: ImpulseWave       # найденная волновая структура
    reason: str             # текстовое описание
    confidence: float       # уверенность 0.0–1.0
    created_at: str
```

#### Правила генерации BUY-сигнала (восходящий импульс)

```
Условие 1 (обязательное):
  Найден завершённый восходящий импульс (5 волн, confidence >= 0.7)
  Импульс завершился не более N свечей назад (N = lookback * 3)

Условие 2 (обязательное):
  После завершения импульса наблюдается откат (коррекция)
  Текущая цена > p4.price (не ушли ниже дна волны 4)

Условие 3 (обязательное — подтверждение пробоем):
  Текущая close > p3.price (пробой вершины волны 3)
  ИЛИ current_price > p1.price (пробой вершины волны 1 — более ранний вход)

Условие 4 (опциональное — RSI-дивергенция):
  Если RSI_DIVERGENCE_ENABLED:
    RSI на p5 < RSI на p3 (медвежья дивергенция — подтверждение окончания)

Уровни:
  entry_price = текущая close (или лимит чуть выше)
  stop_loss   = p4.price - buffer  (где buffer = ATR * 0.5)
  take_profit = entry + (p5.price - p0.price) * 1.618  (экстраполяция)
```

#### Правила SELL-сигнала — зеркально для нисходящего импульса.

### 8.2. Риск-менеджмент

**Файл:** `risk/manager.py`

#### Расчёт размера позиции

```python
def calculate_position_size(capital: float, entry: float,
                             stop_loss: float, lot_size: int,
                             min_price_step: float) -> int:
    """
    Возвращает количество лотов.

    Формула:
      risk_amount = capital * RISK_PERCENT
      price_risk_per_lot = abs(entry - stop_loss) * lot_size
      slippage_per_lot = SLIPPAGE_TICKS * min_price_step * lot_size
      total_risk_per_lot = price_risk_per_lot + slippage_per_lot
      lots = floor(risk_amount / total_risk_per_lot)

    Проверки:
      lots >= 1 (минимум 1 лот)
      lots * entry * lot_size <= capital * MAX_CAPITAL_PER_TRADE
    """
```

**Пример (SBER, lot_size=10):**
```
capital = 500,000 руб.
entry = 300.00 руб.
stop_loss = 285.00 руб. (=-15 руб.)
risk_amount = 500,000 * 0.02 = 10,000 руб.
price_risk_per_lot = 15 * 10 = 150 руб./лот
lots = floor(10,000 / 150) = 66 лотов = 660 акций
Проверка: 66 * 300 * 10 = 198,000 < 100,000 (20%) ✗ → ограничить
MAX лотов = floor(500,000 * 0.20 / (300 * 10)) = 33 лота
Итог: 33 лота
```

#### Circuit Breaker

```python
class CircuitBreaker:
    """
    Останавливает торговлю при серии убытков.
    При срабатывании: уведомление в MAX, статус STOPPED, ждёт ручного сброса.
    """
    consecutive_losses: int = 0
    threshold: int = CIRCUIT_BREAKER_LOSSES  # дефолт: 3
    is_open: bool = False   # True = торговля остановлена

    def on_trade_closed(self, pnl: float):
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.threshold:
                self.is_open = True  # СТОП
                notify_critical("Circuit breaker сработал!")
        else:
            self.consecutive_losses = 0
```

### 8.3. Ресурсы этапа 4

- RSI дивергенция объяснение: https://www.investopedia.com/terms/d/divergence.asp
- Фибоначчи-проекции (тейк-профит): https://www.investopedia.com/terms/f/fibonacciretracement.asp
- Kelly Criterion (альтернативный расчёт позиции, для справки): https://www.investopedia.com/terms/k/kellycriterion.asp
- Управление капиталом: https://www.investopedia.com/articles/trading/09/money-management-techniques.asp

---

## 9. Этап 5: Исполнение сделок и стоп-лосс

### 9.1. OrderExecutor

**Файл:** `execution/executor.py`

#### Жизненный цикл заявки

```python
async def execute_signal(signal: Signal, position_size: int) -> None:
    """
    1. Выставить лимитную заявку (entry_price)
    2. Ждать исполнения MAX 30 секунд
    3. Если не исполнена — отменить, выставить рыночную
    4. При исполнении — активировать мониторинг стоп-лосса
    5. Записать в trades (status='OPEN')
    6. Уведомить в MAX: «Позиция открыта»
    """
```

#### Программный стоп-лосс

```python
async def monitor_stop_loss(trade: Trade) -> None:
    """
    Подписывается на тики symbol через WebSocket.
    При каждом тике:
      if trade.side == 'BUY' and tick.price <= trade.stop_loss:
          await close_position(trade, reason='stop_loss')
      if trade.side == 'BUY' and tick.price >= trade.take_profit:
          await close_position(trade, reason='take_profit')
    """
```

#### Аварийное закрытие при потере WebSocket

```python
async def emergency_close_all() -> None:
    """
    Вызывается при: потере WS соединения после max retry.
    Для каждой открытой позиции:
      - Выставить рыночную заявку на полное закрытие
      - Записать в БД с exit_reason='emergency_close'
    Уведомить в MAX: «АВАРИЙНОЕ ЗАКРЫТИЕ ВСЕХ ПОЗИЦИЙ»
    """
```

### 9.2. Ресурсы этапа 5

- asyncio Event Loop и задачи: https://docs.python.org/3/library/asyncio-eventloop.html
- asyncio.Task и отмена задач: https://docs.python.org/3/library/asyncio-task.html
- WebSocket в Python (websockets lib): https://websockets.readthedocs.io/
- aiohttp WebSocket: https://docs.aiohttp.org/en/stable/client_websocket.html

---

## 10. Этап 6: Отчётность и уведомления

### 10.1. Reporter

**Файл:** `reporting/reporter.py`

#### Финансовые метрики

```python
def calculate_metrics(trades: list[Trade]) -> dict:
    """
    Рассчитывает:
    - total_return_pct  : суммарная доходность (%)
    - max_drawdown_pct  : максимальная просадка (%)
    - sharpe_ratio      : коэффициент Шарпа (безрисковая ставка = 16% годовых для России)
    - win_rate          : % прибыльных сделок
    - avg_win           : средняя прибыль по выигрышным сделкам
    - avg_loss          : средний убыток по проигрышным сделкам
    - profit_factor     : gross_profit / gross_loss
    - total_trades      : количество сделок
    - avg_trade_duration: средняя длительность сделки
    """
```

**Equity Curve:**
- Строится по DataFrame: `[date, capital]`
- Рисуется через Plotly как линейный график
- Отображается в веб-интерфейсе

**Экспорт в Excel:**
- Лист 1: Метрики
- Лист 2: Список сделок
- Лист 3: Equity curve (как данные + встроенный график)
- Библиотека: `openpyxl`

### 10.2. Уведомления в MAX (мессенджер)

**Файл:** `notifications/max_bot.py`
**Библиотека:** `maxapi` (https://pypi.org/project/maxapi/)
**Документация API:** https://dev.max.ru/docs-api

#### Шаблоны уведомлений

```
🟢 СИГНАЛ: ПОКУПКА {SYMBOL}
Вход: {entry} руб.
Стоп: {sl} руб. ({sl_pct}%)
Цель: {tp} руб. ({tp_pct}%)
Размер: {lots} лот(ов) = {qty} акций
Волна: {confidence:.0%} уверенность

📈 ПОЗИЦИЯ ОТКРЫТА: {SYMBOL}
Исполнено: {filled_price} руб. × {qty} акций

🔴 ПОЗИЦИЯ ЗАКРЫТА: {SYMBOL}
P&L: {pnl:+.0f} руб. ({pnl_pct:+.1f}%)
Причина: {exit_reason}

🛑 CIRCUIT BREAKER: торговля остановлена
{N} убыточных сделок подряд. Требуется ревью.

⚠️ ОШИБКА: {module}
{message}
```

#### Реализация через maxapi

```python
from maxapi import Bot

bot = Bot(token=MAX_BOT_TOKEN)

async def send_notification(message: str, user_id: int) -> None:
    await bot.send_message(user_id=user_id, text=message)
```

### 10.3. Ресурсы этапа 6

- maxapi Python (GitHub): https://github.com/max-messenger/max-botapi-python
- maxapi на PyPI: https://pypi.org/project/maxapi/
- MAX API документация: https://dev.max.ru/docs-api
- openpyxl документация: https://openpyxl.readthedocs.io/en/stable/
- Max Drawdown формула: https://www.investopedia.com/terms/m/maximum-drawdown.asp
- Sharpe Ratio с безрисковой ставкой РФ: https://www.investopedia.com/terms/s/sharperatio.asp

---

## 11. Этап 7: Веб-интерфейс

**Технологии:** FastAPI + Jinja2 + Plotly.js + SSE для real-time обновлений.

### 11.1. Страницы и роутеры

#### Дашборд `/` (`web/routers/dashboard.py`)
- Текущий баланс и стоимость портфеля
- Открытые позиции (таблица)
- Последние 5 сигналов
- Кнопки: «Запустить робота» / «Остановить»
- Статус подключения к БКС (зелёная/красная точка)
- Circuit breaker статус

#### График `/chart/{symbol}` (`web/routers/charts.py`)
- Интерактивный candlestick (Plotly.js)
- Переключатель таймфрейма (1H / 4H / 1D)
- Нанесённые свинг-точки (синие маркеры)
- Нанесённые волны W1-W5 (цветные линии)
- Уровни активных позиций (entry/sl/tp)

#### Журнал сделок `/trades` (`web/routers/trades.py`)
- Таблица всех сделок: дата, инструмент, направление, вход, выход, P&L
- Фильтры: по инструменту, по датам, только открытые
- Кнопка ручного закрытия позиции
- Кнопка экспорта в Excel

#### Отчёты `/reports`
- Equity curve
- Таблица метрик
- Сравнение бэктест vs реальная торговля

#### Настройки `/settings` (`web/routers/settings.py`)
- Форма изменения параметров стратегии (RISK_PERCENT, MAX_POSITIONS и т.д.)
- Список торгуемых инструментов (чекбоксы)
- Сохранение в `config.py`

### 11.2. Real-time обновления (SSE)

```python
# web/app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio, json

app = FastAPI()

@app.get("/sse/updates")
async def sse_updates():
    async def event_generator():
        while True:
            data = await get_latest_state()
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(5)  # обновление каждые 5 секунд
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

На фронтенде: `EventSource('/sse/updates')` → обновление дашборда без перезагрузки.

### 11.3. Ресурсы этапа 7

- FastAPI документация: https://fastapi.tiangolo.com/
- Jinja2 шаблоны: https://jinja.palletsprojects.com/en/stable/
- Server-Sent Events (SSE): https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- Plotly.js candlestick: https://plotly.com/javascript/candlestick-charts/
- Plotly Python (для серверной генерации): https://plotly.com/python/
- FastAPI SSE пример: https://fastapi.tiangolo.com/advanced/custom-response/

---

## 12. Требования к безопасности

1. **Токены и секреты:** только в `.env`, никогда в коде или логах. `.env` добавлен в `.gitignore`.
2. **Шифрование токенов в памяти:** использовать `cryptography.fernet` для дополнительной защиты при сохранении refresh token.
3. **Логи:** токены, пароли, приватные данные никогда не попадают в лог-файлы.
4. **Веб-интерфейс:** запускается только на `127.0.0.1` (localhost), НЕ на внешний интерфейс. Базовая HTTP-авторизация если требуется доступ извне.
5. **SQL-инъекции:** все запросы через параметризованные statements (SQLAlchemy / aiosqlite).
6. **Резервные копии БД:** ежедневное копирование `investor.db` в `backups/investor_YYYYMMDD.db`.

---

## 13. Стратегия тестирования

### Unit-тесты (`tests/`)

| Файл теста | Что тестируется | Приоритет |
|---|---|---|
| `test_swing_detector.py` | Поиск свингов на синтетических данных | Критический |
| `test_wave_analyzer.py` | Проверка 5-волновых структур, правила Эллиотта | Критический |
| `test_risk_manager.py` | Расчёт позиции, circuit breaker | Критический |
| `test_backtest_engine.py` | Метрики, корректность бэктеста | Высокий |
| `test_indicators.py` | ATR, RSI расчёты | Высокий |
| `test_signal_generator.py` | Генерация сигналов на тестовых волнах | Высокий |

**Запуск:**
```bash
pytest tests/ -v --cov=. --cov-report=html
```

**Цель:** покрытие кода ≥ 70% для критических модулей (`analysis/`, `risk/`).

### Интеграционные тесты

- Тест подключения к БКС API (с реальными учётными данными, на демо-окружении).
- Тест полного цикла: загрузка данных → анализ → сигнал → заявка (на демо-счёте).

### Нагрузочное тестирование

- Непрерывная работа в течение 24 часов на исторических данных (прокрутка истории).
- Проверка: нет утечек памяти, нет зависаний, логи чистые.

### Ресурсы для тестирования

- `pytest` документация: https://docs.pytest.org/en/stable/
- `pytest-asyncio`: https://pytest-asyncio.readthedocs.io/en/latest/
- `pytest-cov`: https://pytest-cov.readthedocs.io/en/latest/

---

## 14. Глоссарий

| Термин | Объяснение |
|---|---|
| **Свеча (candle)** | Данные за период: open, high, low, close, volume |
| **Волна Эллиотта** | Ценовое движение, классифицируемое как часть 5-волнового импульса или 3-волновой коррекции |
| **Импульс (impulse)** | 5-волновая структура в направлении основного тренда |
| **Свинг-точка (swing)** | Значимый локальный максимум или минимум цены |
| **ATR** | Average True Range — индикатор волатильности |
| **RSI** | Relative Strength Index — индикатор силы тренда (0-100) |
| **Дивергенция RSI** | Цена обновляет максимум, RSI — нет. Сигнал ослабления |
| **Фибоначчи** | Числа 0.382, 0.618, 1.618 — соотношения длин волн |
| **Стоп-лосс** | Уровень принудительного закрытия убыточной позиции |
| **Тейк-профит** | Уровень фиксации прибыли |
| **Лот** | Минимальная единица торговли (SBER: 10 акций = 1 лот) |
| **Бэктестинг** | Проверка стратегии на исторических данных |
| **Просадка (drawdown)** | Падение капитала от максимума |
| **Шарп (Sharpe)** | Доходность/риск. Хорошо > 1.5 |
| **Circuit breaker** | Автоматическая остановка торговли при серии убытков |
| **MAX** | Российский мессенджер (VK), используется для уведомлений |
| **MOEX** | Московская биржа |
| **БКС** | Брокер, через которого выставляются заявки |
| **P&L** | Profit & Loss — прибыль и убытки |

---

*Документ является живым — обновляется при изменении требований. Последнее обновление: 21.03.2026*
