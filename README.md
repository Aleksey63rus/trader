# Elliott Wave Trader

Алготрейдер на основе волновой теории Эллиотта с интеграцией БКС Trade API.

## Архитектура

- **Async-first**: весь I/O через `asyncio`/`aiohttp`, FastAPI
- **5-волновой импульс = 6 свинг-точек**: `low₀→high₁→low₂→high₃→low₄→high₅`
- **Машина состояний сделки**: `IDLE→SIGNAL→ORDER_PLACED→OPEN→CLOSING→CLOSED`

## Установка

```powershell
cd c:\investor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dev-инструменты и pre-commit

```powershell
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

Если `pre-commit` внёс автофиксы, выполните повторно `git add .` и `pre-commit run --all-files` до полностью зелёного результата.

## Этап 1: Бэктестинг (текущий)

### Запуск веб-интерфейса

```powershell
python main.py
# открыть http://127.0.0.1:8000
```

### Загрузка исторических данных

Скачайте CSV с [Финам](https://www.finam.ru/profile/moex-akcii/sberbank/export/):
- Тикер: SBER
- Таймфрейм: 1 час
- Период: 2023–2024

### Запуск тестов

```powershell
python -m pytest tests -v
```

Ожидается **27** успешных тестов. Итоговый отчёт: **[REPORT.md](REPORT.md)**.

## Структура проекта

```
c:\investor\
├── config.py               # все параметры стратегии
├── main.py                 # точка входа
├── analysis/
│   ├── indicators.py       # ATR, RSI
│   ├── swing_detector.py   # 6-точечные свинги с ATR-фильтром
│   └── wave_analyzer.py    # 5-волновые импульсы + Фибоначчи
├── backtesting/
│   └── engine.py           # бэктест с комиссиями и метриками
├── data/
│   └── db.py               # SQLite (candles, signals, trades, logs)
├── web/
│   ├── app.py              # FastAPI
│   └── templates/
│       └── index.html      # Plotly-дашборд
└── tests/
    ├── test_swing.py
    └── test_wave.py
```

## Правила Эллиотта (реализованы)

**Обязательные:**
1. Волна 2 не откатывает более 100% волны 1
2. Волна 3 не самая короткая среди волн 1, 3, 5
3. Волна 4 не входит в ценовую зону волны 1

**Фибоначчи (влияют на `confidence_score`):**
4. Откат волны 2: 50–78.6% волны 1
5. Расширение волны 3: ~161.8% волны 1
6. Откат волны 4: 23.6–50% волны 3
7. Волна 5 ≈ волна 1 (или 61.8% от p0→p3)

## Конфигурация

Все параметры в `config.py`:

| Параметр | Значение | Описание |
|---|---|---|
| `LOOKBACK` | 5 | Окно поиска свинг-точек |
| `ATR_MULTIPLIER` | 1.5 | Минимальный размах свинга |
| `FIB_TOLERANCE` | 10% | Допуск соотношений Фибоначчи |
| `RISK_PERCENT` | 2% | Риск на сделку |
| `COMMISSION_RATE` | 0.05% | Комиссия БКС |

## Следующие этапы

- **Этап 2**: Подключение к БКС (BCSPy, WebSocket, кэш в SQLite)
- **Этап 3**: Генератор сигналов + риск-менеджмент
- **Этап 4**: Исполнение заявок + программный стоп-лосс
- **Этап 5**: Уведомления в MAX (maxapi)
- **Этап 6**: Полный веб-дашборд с SSE
