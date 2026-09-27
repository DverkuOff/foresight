# Форсайт (Foresight) · предиктор задержек городского транспорта

Foresight принимает поток телеметрии NDTP, восстанавливает проходы остановок, за 10–15 минут до планового
прибытия прогнозирует задержку (с интервалом P10–P90 и вероятностью опоздания), поднимает алерты с причиной и
показывает всё диспетчеру на дашборде. Каждый прогноз затем сверяется с фактом на потоке.

- **Жюри:** начните с [инструкции](guide/jury-guide.md) — запуск одной командой, где смотреть прогнозы и алерты.
- **API:** [спецификации OpenAPI всех сервисов (Swagger)](api/index.html); на стенде — `/docs` у каждого сервиса.
- **Код:** справочник модулей backend, ML, общих библиотек и replayer (ниже).

```{toctree}
:caption: Руководства
:maxdepth: 1

guide/jury-guide
guide/performance
guide/architecture
guide/api-contract
guide/online-validation
guide/ml-report
guide/observability
guide/third-party
```

```{toctree}
:caption: Справочник по коду
:maxdepth: 2

reference/backend
reference/ml
reference/shared
reference/replayer
```
