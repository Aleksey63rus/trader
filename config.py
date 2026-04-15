import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Торгуемые инструменты
SYMBOLS = ["SBER", "GAZP"]

# Таймфрейм
INTERVAL = "1H"

# История для загрузки при старте (дней)
HISTORY_DAYS = 90

# Параметры ATR
ATR_PERIOD = 14
ATR_MULTIPLIER = 0.5  # минимальный размах свинга = ATR_MULTIPLIER × ATR14

# Детектор свингов
LOOKBACK = 7  # окно поиска локальных экстремумов (свечей)

# Волновой анализ
FIB_TOLERANCE = 0.25  # допуск ±25% для соотношений Фибоначчи

# Риск-менеджмент
RISK_PERCENT = 0.02  # риск на сделку от капитала
MAX_POSITIONS = 3
CIRCUIT_BREAKER_LOSSES = 3  # стоп после N убытков подряд
MAX_POSITION_PCT = 0.20  # не более 20% капитала в одной позиции

# RSI
RSI_PERIOD = 14

# Бэктестинг
COMMISSION_RATE = 0.0005  # 0.05% за сделку (БКС)
SLIPPAGE_TICKS = 1  # проскальзывание в тиках

# База данных
DB_PATH = BASE_DIR / "data" / "trader.db"

# Веб-интерфейс
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
APP_HOST = os.getenv("APP_HOST", WEB_HOST)
APP_PORT = int(os.getenv("APP_PORT", str(WEB_PORT)))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
