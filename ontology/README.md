# Диплом 2 — Система збору даних та знань засобами ШІ

## Що це

Онтологія (OWL/RDF) + SPARQL-endpoint (Fuseki) + HTTP API, через який
інші дипломи запитують характеристики обладнання. Плюс ШІ-компонент,
який парсить PDF-паспорти через LLM і автоматично наповнює онтологію.

## Компоненти

| Файл | Призначення |
|---|---|
| `equipment.ttl` | Онтологія в Turtle: класи, властивості, екземпляри |
| `load_ontology.py` | Завантажує .ttl у Fuseki (раз після старту) |
| `ontology_api.py` | Flask-обгортка над SPARQL, JSON API для інших дипломів |
| `llm_parser.py` | LLM витягує характеристики з PDF → Turtle |

## Запуск

```bash
pip install -r requirements.txt
python load_ontology.py        # одноразово
python ontology_api.py         # HTTP сервер на :5000
```

## Редагування онтології

Зручно редагувати `equipment.ttl` у [Protégé](https://protege.stanford.edu/).
Відкрити → внести зміни → зберегти як Turtle → `python load_ontology.py`.

## API

| Маршрут | Що повертає |
|---|---|
| `GET /devices` | Список усіх зареєстрованих пристроїв |
| `GET /device/<id>/specs` | Усі літеральні властивості пристрою |
| `GET /device/<id>/expected-bounds` | Межі для детектора аномалій (Диплом 3) |
| `GET /device/<id>/components` | Компоненти обладнання |

## ШІ-парсер паспортів

```bash
# варіант А: OpenAI (поклади OPENAI_API_KEY у .env)
python llm_parser.py ecodan_02 datasheets/ehst20d.pdf

# варіант Б: локальний Ollama
ollama pull llama3
python llm_parser.py ecodan_02 datasheets/ehst20d.pdf
```

Вивід можна додати в `equipment.ttl` і перезавантажити в Fuseki.
