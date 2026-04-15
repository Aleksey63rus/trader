"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    INVESTOR — СПРАВОЧНИК ПРОЕКТА                           ║
║              Файл для AI и разработчиков: читать ПЕРВЫМ                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ПРОЕКТ: Автоматическая торговая система на основе волновой теории Эллиотта ║
║  БРОКЕР: БКС (Россия, MOEX)                                                ║
║  БИРЖА:  ММВБ / Московская биржа (MOEX)                                    ║
║  ЯЗЫК:   Python 3.9+                                                       ║
║  ВЕРСИЯ: 1.0 (в разработке)                                                ║
║  ТЗ:     c:\\investor\\full_tz.md  ← ГЛАВНЫЙ ДОКУМЕНТ ПРОЕКТА               ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
КАК ПОЛЬЗОВАТЬСЯ ЭТИМ ФАЙЛОМ (для AI):

1. Прочитать раздел PROJECT_SUMMARY — понять суть проекта.
2. Прочитать FILE_MAP — найти нужный файл для задачи.
3. Прочитать MODULE_RESPONSIBILITIES — понять, за что отвечает каждый модуль.
4. Использовать константы из этого файла напрямую (не дублировать в других файлах).
5. Секреты (токены) — ТОЛЬКО в .env, никогда в этом файле.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# КРАТКОЕ ОПИСАНИЕ ПРОЕКТА
# =============================================================================

PROJECT_SUMMARY = """
INVESTOR — десктопная торговая система (FastAPI-сервер + браузер).

ЧТО ДЕЛАЕТ:
  1. Подключается к БКС Trade API, загружает свечи по акциям MOEX.
  2. Ищет 5-волновые импульсы Эллиотта (6 опорных точек: p0..p5).
  3. Генерирует сигналы BUY/SELL при обнаружении завершённых волн.
  4. Рассчитывает размер позиции с учётом риска 2% на сделку.
  5. Выставляет заявки, отслеживает стоп-лосс программно (WebSocket).
  6. Ведёт журнал, формирует отчёты, отправляет уведомления в MAX.

ПОРЯДОК РАЗРАБОТКИ:
  1. Этап 1: каркас проекта (структура, конфиг, логирование, БД)
  2. Этап 2: волновой анализ + бэктестинг (БЕЗ брокера)
  3. Этап 3: БКС API + загрузка реальных данных
  4. Этап 4: сигналы + риск-менеджмент
  5. Этап 5: исполнение сделок + стоп-лосс
  6. Этап 6: отчёты + уведомления MAX
  7. Этап 7: веб-интерфейс (дашборд, графики)

КЛЮЧЕВЫЕ РЕШЕНИЯ:
  - Async-first: весь I/O через asyncio, FastAPI (не Flask)
  - 6 свинг-точек для импульса (НЕ 10 — ошибка в первичном ТЗ исправлена)
  - Стоп-лосс программный (API БКС не поддерживает стоп-заявки)
  - Circuit breaker: остановка после 3 убытков подряд
  - Уведомления: MAX мессенджер (maxapi)
  - БД: SQLite → возможна миграция на PostgreSQL
"""

# =============================================================================
# КАРТА ФАЙЛОВ ПРОЕКТА
# =============================================================================
# Формат: 'путь': 'описание файла и его назначение'

FILE_MAP = {
    # --- Корневые файлы ---
    'full_tz.md':           'ГЛАВНЫЙ ДОКУМЕНТ: полное ТЗ, все правила, алгоритмы, ресурсы',
    'constants.py':         'ЭТОТ ФАЙЛ: справочник проекта, карта файлов, все константы',
    'config.py':            'Параметры стратегии (ATR, риск, Фибоначчи). Без секретов.',
    'logging_config.py':    'Настройка логирования (файл logs/investor.log + консоль)',
    'main.py':              'Точка входа. Запускает оркестратор и FastAPI сервер.',
    '.env':                 'СЕКРЕТЫ: BCS_CLIENT_ID, BCS_CLIENT_SECRET, MAX_BOT_TOKEN. Не в git!',
    '.env.example':         'Шаблон .env без значений. Копировать → заполнять.',
    'requirements.txt':     'Список Python-зависимостей с версиями.',
    'README.md':            'Инструкция по установке и запуску.',

    # --- Брокер ---
    'broker/__init__.py':   'Экспорт: BrokerClient (текущая реализация)',
    'broker/base.py':       'Абстрактный класс AbstractBrokerClient (интерфейс)',
    'broker/bcs_client.py': 'Реализация для БКС через BCSPy. OAuth2, REST, WebSocket.',

    # --- Данные ---
    'data/__init__.py':     'Экспорт: DataManager',
    'data/manager.py':      'DataManager: загрузка свечей, кэш в SQLite, ATR расчёт',
    'data/db.py':           'SQLite: создание таблиц, CRUD операции, миграции',
    'data/investor.db':     'SQLite база данных (создаётся автоматически при первом запуске)',

    # --- Анализ ---
    'analysis/__init__.py':      'Экспорт: WaveAnalyzer, SwingDetector',
    'analysis/indicators.py':    'Расчёт ATR14 и RSI14 через pandas/numpy',
    'analysis/swing_detector.py':'Поиск свинг-точек с ATR-фильтром и lookback-окном',
    'analysis/wave_analyzer.py': 'Идентификация 5-волновых импульсов (6 точек: p0..p5)',

    # --- Сигналы ---
    'signals/__init__.py':   'Экспорт: SignalGenerator, Signal',
    'signals/generator.py':  'Генерация BUY/SELL сигналов на основе волн. Dataclass Signal.',

    # --- Риск ---
    'risk/__init__.py':      'Экспорт: RiskManager, CircuitBreaker',
    'risk/manager.py':       'Расчёт размера позиции (лоты), circuit breaker, лимиты',

    # --- Исполнение ---
    'execution/__init__.py': 'Экспорт: OrderExecutor',
    'execution/executor.py': 'Выставление заявок, программный стоп-лосс, аварийное закрытие',

    # --- Отчёты ---
    'reporting/__init__.py': 'Экспорт: Reporter',
    'reporting/reporter.py': 'Метрики (Sharpe, drawdown, win rate), equity curve, Excel',

    # --- Уведомления ---
    'notifications/__init__.py': 'Экспорт: Notifier',
    'notifications/max_bot.py':  'Уведомления в MAX мессенджер через maxapi',

    # --- Бэктестинг ---
    'backtesting/__init__.py': 'Экспорт: BacktestEngine',
    'backtesting/engine.py':   'Прогон стратегии на исторических CSV-данных',

    # --- Веб-интерфейс ---
    'web/__init__.py':              'Экспорт: FastAPI app',
    'web/app.py':                   'Главное FastAPI приложение, SSE endpoint',
    'web/routers/dashboard.py':     'Роутер: дашборд (/)',
    'web/routers/charts.py':        'Роутер: графики (/chart/{symbol})',
    'web/routers/trades.py':        'Роутер: журнал сделок (/trades)',
    'web/routers/settings.py':      'Роутер: настройки стратегии (/settings)',
    'web/templates/base.html':      'Базовый HTML-шаблон (Jinja2)',
    'web/templates/dashboard.html': 'Шаблон дашборда',
    'web/templates/chart.html':     'Шаблон страницы графика',
    'web/templates/trades.html':    'Шаблон журнала сделок',
    'web/static/css/main.css':      'Стили интерфейса',
    'web/static/js/main.js':        'JavaScript: SSE клиент, Plotly графики',

    # --- Тесты ---
    'tests/test_swing_detector.py': 'Юнит-тесты поиска свинг-точек',
    'tests/test_wave_analyzer.py':  'Юнит-тесты 5-волновых импульсов',
    'tests/test_risk_manager.py':   'Юнит-тесты расчёта позиции и circuit breaker',
    'tests/test_backtest_engine.py':'Юнит-тесты движка бэктестинга',

    # --- Директории ---
    'logs/':     'Лог-файлы по дням: investor_YYYY-MM-DD.log',
    'backups/':  'Резервные копии БД: investor_YYYYMMDD.db',
}

# =============================================================================
# ОТВЕТСТВЕННОСТИ МОДУЛЕЙ (для AI: что делает каждый модуль)
# =============================================================================

MODULE_RESPONSIBILITIES = {
    'broker':      'Всё взаимодействие с БКС: токены, REST запросы, WebSocket подписки',
    'data':        'Хранение свечей в SQLite, загрузка истории, кэш, расчёт ATR/RSI',
    'analysis':    'Алгоритм Эллиотта: поиск свингов → поиск 5-волн → структура ImpulseWave',
    'signals':     'Из ImpulseWave формирует Signal (entry, sl, tp, direction)',
    'risk':        'Рассчитывает quantity в лотах, проверяет лимиты, управляет circuit breaker',
    'execution':   'Отправляет заявки в БКС, мониторит стоп по WebSocket-тикам',
    'reporting':   'Читает trades из БД, считает метрики, генерирует Excel и Plotly-графики',
    'notifications':'Отправляет сообщения в MAX мессенджер пользователя',
    'backtesting': 'Симулирует торговлю на DataFrame без реальных заявок',
    'web':         'FastAPI: HTML-страницы, SSE обновления, API для JS-клиента',
}

# =============================================================================
# КОНСТАНТЫ БАЗЫ ДАННЫХ
# =============================================================================

DB_PATH = 'data/investor.db'

DB_TABLES = {
    'candles':     'Исторические свечи: symbol, interval, dt, OHLCV, atr14',
    'signals':     'Сгенерированные сигналы: entry, sl, tp, wave_json, acted_on',
    'trades':      'Сделки: вход, выход, P&L, комиссия, exit_reason, status',
    'app_logs':    'Логи уровня ERROR+ в БД для отображения в веб-интерфейсе',
    'instruments': 'Справочник инструментов: symbol, lot_size, min_price_step',
}

DB_TABLE_COLUMNS = {
    'candles': ['id', 'symbol', 'interval', 'dt', 'open', 'high', 'low',
                'close', 'volume', 'atr14'],
    'signals': ['id', 'symbol', 'created_at', 'direction', 'entry_price',
                'stop_loss', 'take_profit', 'wave_json', 'reason',
                'acted_on', 'backtest_run'],
    'trades':  ['id', 'signal_id', 'symbol', 'side', 'lot_size', 'quantity',
                'entry_price', 'entry_time', 'exit_price', 'exit_time',
                'commission', 'pnl', 'pnl_pct', 'exit_reason', 'status'],
    'app_logs':['id', 'created_at', 'level', 'module', 'message'],
    'instruments': ['symbol', 'name', 'lot_size', 'min_price_step',
                    'currency', 'is_active'],
}

# =============================================================================
# КОНСТАНТЫ ВОЛНОВОГО АНАЛИЗА
# =============================================================================

# ВАЖНО: 5-волновой импульс = 6 точек (p0..p5), НЕ 10!
# p0=начало, p1=конец W1, p2=конец W2, p3=конец W3, p4=конец W4, p5=конец W5
IMPULSE_POINT_NAMES = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
IMPULSE_POINT_COUNT = 6

# Тип чередования точек для ВОСХОДЯЩЕГО импульса:
IMPULSE_UP_SWING_TYPES   = ['low', 'high', 'low', 'high', 'low', 'high']
# Тип чередования точек для НИСХОДЯЩЕГО импульса:
IMPULSE_DOWN_SWING_TYPES = ['high', 'low', 'high', 'low', 'high', 'low']

# Обязательные правила Эллиотта (нарушение = структура отклоняется)
ELLIOTT_RULES = {
    'R1_wave2_above_start':   'p2.price > p0.price (W2 не ниже начала)',
    'R2_wave3_not_shortest':  'len(W3) > min(len(W1), len(W5))',
    'R3_wave4_no_overlap':    'p4.price > p1.price (W4 не ниже вершины W1)',
    'R4_impulse_direction':   'p1>p0, p3>p2, p5>p4 (все волны идут вверх)',
}

# Уровни Фибоначчи
FIB_LEVELS = {
    'wave2_retrace': (0.382, 0.618),   # W2 = 38.2%–61.8% от W1
    'wave3_ext':     (1.618, 2.618),   # W3 = 161.8%–261.8% от W1
    'wave4_retrace': (0.382, 0.618),   # W4 = 38.2%–61.8% от W3
    'wave5_ext':     (0.618, 1.618),   # W5 = 61.8%–161.8% от W1
    'tp_extension':  1.618,            # тейк-профит = 161.8% от всего импульса
}

FIB_TOLERANCE = 0.10  # допуск ±10% для проверки Фибоначчи

# Порог уверенности для принятия волновой структуры (0.0–1.0)
MIN_CONFIDENCE_THRESHOLD = 0.70

# =============================================================================
# КОНСТАНТЫ ИНДИКАТОРОВ
# =============================================================================

ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5      # мин. размах свинга = ATR * ATR_MULTIPLIER
LOOKBACK_WINDOW = 5       # окно (в свечах) для поиска локальных экстремумов

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_DIVERGENCE_ENABLED = True  # опциональный фильтр в генераторе сигналов

# =============================================================================
# КОНСТАНТЫ РИСК-МЕНЕДЖМЕНТА
# =============================================================================

RISK_PERCENT = 0.02        # 2% капитала риска на сделку
MAX_POSITIONS = 3          # максимум открытых позиций одновременно
MAX_CAPITAL_PER_TRADE = 0.20   # не более 20% капитала в одной позиции
SLIPPAGE_TICKS = 2         # допустимое проскальзывание в тиках
CIRCUIT_BREAKER_LOSSES = 3 # остановить торговлю после N убытков подряд
STOP_BUFFER_ATR = 0.5      # стоп = p4 - ATR * STOP_BUFFER_ATR

# =============================================================================
# КОНСТАНТЫ БРОКЕРА И КОМИССИЙ
# =============================================================================

COMMISSION_PERCENT = 0.0005   # 0.05% за сделку (тариф БКС)

# Известные лот-сайзы инструментов MOEX (загружаются из БД; здесь как fallback)
DEFAULT_LOT_SIZES = {
    'SBER':  10,    # Сбербанк
    'GAZP':  10,    # Газпром
    'LKOH':   1,    # Лукойл
    'YNDX':   1,    # Яндекс
    'GMKN':   1,    # Норильский никель
    'ROSN':   1,    # Роснефть
    'NVTK':   1,    # НоваТЭК
    'TATN':  10,    # Татнефть
    'MTSS':  10,    # МТС
    'MGNT':   1,    # Магнит
}

# Торгуемые инструменты по умолчанию
DEFAULT_SYMBOLS = ['SBER', 'GAZP', 'LKOH', 'YNDX', 'GMKN']

# Доступные таймфреймы (коды БКС Trade API)
SUPPORTED_INTERVALS = ['1M', '5M', '15M', '1H', '4H', '1D']
DEFAULT_INTERVAL = '1H'

# Глубина загружаемой истории в днях
HISTORY_DAYS = 120

# =============================================================================
# КОНСТАНТЫ УВЕДОМЛЕНИЙ (MAX МЕССЕНДЖЕР)
# =============================================================================

# Токен и user_id берутся из .env (MAX_BOT_TOKEN, MAX_USER_ID)
MAX_NOTIFICATION_TYPES = {
    'SIGNAL':          '🟢 Новый сигнал',
    'POSITION_OPENED': '📈 Позиция открыта',
    'POSITION_CLOSED': '📊 Позиция закрыта',
    'STOP_LOSS_HIT':   '🔴 Стоп-лосс сработал',
    'TAKE_PROFIT_HIT': '✅ Тейк-профит достигнут',
    'CIRCUIT_BREAKER': '🛑 CIRCUIT BREAKER: торговля остановлена',
    'WS_DISCONNECTED': '⚠️ WebSocket отключён — позиции под риском',
    'EMERGENCY_CLOSE': '🚨 АВАРИЙНОЕ ЗАКРЫТИЕ всех позиций',
    'ERROR':           '❌ Ошибка системы',
    'STARTED':         '▶️ Робот запущен',
    'STOPPED':         '⏹️ Робот остановлен',
}

# =============================================================================
# КОНСТАНТЫ СОСТОЯНИЙ СДЕЛКИ
# =============================================================================

TRADE_STATUS = {
    'IDLE':           'Нет активности',
    'SIGNAL_PENDING': 'Сигнал ожидает исполнения',
    'ORDER_PLACED':   'Заявка выставлена',
    'OPEN':           'Позиция открыта',
    'CLOSING':        'Идёт закрытие позиции',
    'CLOSED':         'Позиция закрыта',
}

TRADE_SIDES = ('BUY', 'SELL')
ORDER_TYPES  = ('MARKET', 'LIMIT')

EXIT_REASONS = {
    'stop_loss':      'Сработал стоп-лосс',
    'take_profit':    'Достигнут тейк-профит',
    'manual':         'Ручное закрытие пользователем',
    'signal':         'Противоположный сигнал',
    'emergency_close':'Аварийное закрытие (потеря WebSocket)',
    'circuit_breaker':'Circuit breaker остановил торговлю',
}

# =============================================================================
# КОНСТАНТЫ ФИНАНСОВЫХ МЕТРИК
# =============================================================================

# Безрисковая ставка для расчёта коэффициента Шарпа (ключевая ставка ЦБ РФ)
RISK_FREE_RATE_ANNUAL = 0.16  # 16% годовых (обновлять при изменении КС)
TRADING_DAYS_PER_YEAR = 252   # рабочих дней на MOEX в году

# Минимальный Sharpe для «хорошей» стратегии
GOOD_SHARPE_THRESHOLD = 1.5

# =============================================================================
# КОНСТАНТЫ ВЕБ-ИНТЕРФЕЙСА
# =============================================================================

APP_HOST = '127.0.0.1'   # только localhost (безопасность)
APP_PORT = 8000
APP_TITLE = 'INVESTOR — Elliott Wave Trader'

# SSE: интервал обновления дашборда в секундах
SSE_UPDATE_INTERVAL = 5

# Маршруты FastAPI
ROUTES = {
    '/':               'Дашборд (текущие позиции, баланс, последние сигналы)',
    '/chart/{symbol}': 'График с волновой разметкой',
    '/trades':         'Журнал всех сделок',
    '/reports':        'Equity curve и финансовые метрики',
    '/settings':       'Настройки стратегии',
    '/backtest':       'Запуск бэктеста',
    '/sse/updates':    'Server-Sent Events: real-time обновления дашборда',
    '/api/close/{id}': 'API: ручное закрытие позиции',
    '/api/start':      'API: запуск робота',
    '/api/stop':       'API: остановка робота',
}

# =============================================================================
# КОНСТАНТЫ ЛОГИРОВАНИЯ
# =============================================================================

LOG_DIR = 'logs'
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB на файл
LOG_BACKUP_COUNT = 30               # хранить 30 файлов (30 дней)

# Модули логирования (используются как имена логгеров)
LOGGERS = {
    'broker':      'investor.broker',
    'data':        'investor.data',
    'analysis':    'investor.analysis',
    'signals':     'investor.signals',
    'risk':        'investor.risk',
    'execution':   'investor.execution',
    'reporting':   'investor.reporting',
    'notifications':'investor.notifications',
    'backtesting': 'investor.backtesting',
    'web':         'investor.web',
    'main':        'investor.main',
}

# =============================================================================
# КОНСТАНТЫ БЭКТЕСТИНГА
# =============================================================================

BACKTEST_DATA_DIR = 'data/backtest_csv'    # папка для CSV-файлов истории
BACKTEST_RESULTS_DIR = 'data/backtest_results'  # результаты в JSON/Excel

# Формат CSV: datetime,open,high,low,close,volume
BACKTEST_CSV_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']
BACKTEST_CSV_DT_FORMAT = '%Y-%m-%d %H:%M:%S'

# Источник исторических данных для бэктеста (скачать вручную):
BACKTEST_DATA_SOURCE = 'https://www.finam.ru/profile/moex-akcii/sberbank/export/'

# =============================================================================
# ВАЖНЫЕ ВНЕШНИЕ РЕСУРСЫ (ссылки из ТЗ)
# =============================================================================

EXTERNAL_RESOURCES = {
    'full_tz':          'c:\\investor\\full_tz.md (ГЛАВНЫЙ ДОКУМЕНТ)',
    'BCSPy_github':     'https://github.com/cia76/BCSPy',
    'BCS_smartlab':     'https://smart-lab.ru/company/bcs/blog/1228126.php',
    'bcs_go_client':    'https://github.com/tigusigalpa/bcs-trade-go',
    'maxapi_pypi':      'https://pypi.org/project/maxapi/',
    'maxapi_github':    'https://github.com/max-messenger/max-botapi-python',
    'max_api_docs':     'https://dev.max.ru/docs-api',
    'fastapi_docs':     'https://fastapi.tiangolo.com/',
    'plotly_candle':    'https://plotly.com/python/candlestick-charts/',
    'finam_export':     'https://www.finam.ru/profile/moex-akcii/sberbank/export/',
    'pandas_docs':      'https://pandas.pydata.org/docs/',
    'asyncio_docs':     'https://docs.python.org/3/library/asyncio.html',
    'aiosqlite_docs':   'https://aiosqlite.omnilib.dev/en/stable/',
    'pytest_docs':      'https://docs.pytest.org/en/stable/',
    'elliott_wave_inv': 'https://www.investopedia.com/articles/technical/111401.asp',
    'atr_investopedia': 'https://www.investopedia.com/terms/a/atr.asp',
    'fibonacci_inv':    'https://www.investopedia.com/articles/technical/04/033104.asp',
}

# =============================================================================
# ИНСТРУКЦИЯ ДЛЯ AI (ЧТО ПРОЧИТАТЬ ПЕРЕД ВЫПОЛНЕНИЕМ ЗАДАЧИ)
# =============================================================================

AI_INSTRUCTIONS = """
При получении задачи в этом проекте:

1. ОПРЕДЕЛИ КАТЕГОРИЮ ЗАДАЧИ:
   - "волны", "эллиотт", "свинг", "импульс" → читай analysis/ + full_tz.md раздел 6
   - "сигнал", "entry", "стоп", "тейк" → читай signals/ + full_tz.md раздел 8
   - "риск", "позиция", "лот", "circuit" → читай risk/ + full_tz.md раздел 8.2
   - "брокер", "API", "заявка", "BCS" → читай broker/ + full_tz.md раздел 7
   - "данные", "свечи", "SQLite", "кэш" → читай data/ + full_tz.md разделы 4,7
   - "бэктест", "история", "CSV", "метрики" → читай backtesting/ + full_tz.md раздел 6.4
   - "уведомление", "MAX", "бот" → читай notifications/ + full_tz.md раздел 10.2
   - "веб", "интерфейс", "дашборд", "график" → читай web/ + full_tz.md раздел 11
   - "отчёт", "Sharpe", "drawdown", "equity" → читай reporting/ + full_tz.md раздел 10.1

2. ОБЯЗАТЕЛЬНО ПРОЧИТАЙ full_tz.md если:
   - Непонятны правила или алгоритм
   - Задача касается архитектуры
   - Нужны правила Эллиотта или Фибоначчи

3. НЕ НАРУШАЙ ЭТИ ПРАВИЛА:
   - Импульс Эллиотта = 6 точек (p0..p5), НИКОГДА не 10
   - Секреты (токены) — только в .env, никогда в коде
   - Весь I/O — async (asyncio), не синхронный
   - Стоп-лосс — только программный через WebSocket-тики
   - При потере WebSocket — АВАРИЙНОЕ закрытие всех позиций

4. СТИЛЬ КОДА:
   - PEP8, type hints везде
   - dataclass для структур данных
   - Логирование через модуль logging (не print)
   - Комментарии только для неочевидной логики
"""
