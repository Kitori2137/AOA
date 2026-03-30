# AOA – Aplikacja Optymalnego Algorytmowania

## Opis projektu

AOA to aplikacja desktopowa napisana w Pythonie, której celem jest wspomaganie analizy danych produkcyjnych, trenowania modeli uczenia maszynowego oraz porównywania wyników różnych metod analitycznych i wizualnych.

Projekt został przygotowany w architekturze modułowej, z wyraźnym podziałem na:

- warstwę `core`, odpowiedzialną za logikę aplikacji, przetwarzanie danych, modele, ewaluację i przygotowanie wyników,
- warstwę `gui`, odpowiedzialną wyłącznie za interfejs użytkownika,
- katalog `tests`, zawierający testy jednostkowe dla warstwy `core`,
- katalog `docs`, zawierający dokumentację użytkową i teoretyczną projektu.

Aplikacja umożliwia między innymi:

- generowanie przykładowych danych produkcyjnych,
- wczytywanie danych z plików CSV,
- trenowanie modeli ML dla jakości, opóźnień i strategii harmonogramowania,
- analizę wyników regresji i klasyfikacji,
- tworzenie wykresów i wizualizacji danych,
- zapis wyników do plików.

Projekt ma charakter rozwojowy i stanowi bazę pod dalszą rozbudowę o nowe algorytmy, dokładniejsze analizy oraz bardziej zaawansowane moduły wspomagania decyzji.

## Aktualna struktura repozytorium

```text
TOOLS/
├── docs/
│   ├── guide.md
│   └── theory.md
├── data/
├── models/
├── src/
│   └── AOA/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── data_generation.py
│       │   ├── data_io.py
│       │   ├── dataset_ops.py
│       │   ├── evaluation.py
│       │   ├── features.py
│       │   ├── models.py
│       │   ├── scheduling.py
│       │   ├── services.py
│       │   ├── sto_models.py
│       │   ├── visualization_service.py
│       │   └── diagrams/
│       │       ├── __init__.py
│       │       ├── correlation_matrix.py
│       │       ├── decision_tree_diagram.py
│       │       ├── gantt_chart.py
│       │       ├── line_chart.py
│       │       └── similarity_matrix.py
│       ├── gui/
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   └── pages/
│       │       ├── __init__.py
│       │       ├── main_page.py
│       │       ├── readme_page.py
│       │       ├── results_page.py
│       │       ├── theory_page.py
│       │       └── visual_page.py
│       └── utils/
│           ├── __init__.py
│           ├── logging_utils.py
│           └── threading_utils.py
├── tests/
│   └── core/
│       ├── test_data_generation.py
│       ├── test_data_io.py
│       ├── test_dataset_ops.py
│       ├── test_features.py
│       ├── test_models.py
│       ├── test_scheduling.py
│       ├── test_services_extra.py
│       ├── test_sto_models.py
│       └── test_visualization_service.py
├── CHANGELOG.md
├── README.md
├── pyproject.toml
└── uv.lock
```



## Segment 3 — uruchomienie projektu przez `uv`


## Jak uruchomić projekt

### Wersja z `uv`

#### 1. Sklonuj repozytorium i przejdź do katalogu projektu

```bash
git clone git@github.com:UZ-FENS/passthebranch-Kitori2137.git
cd passthebranch-Kitori2137
```

#### 2. Zainstaluj zależności i uruchom aplikację
```bash
py -m uv sync --dev
py -m uv run aoa
```

#### 3. Uruchom testy
```bash
py -m uv run pytest
```

---

## Segment 4 — uruchomienie projektu bez `uv`


### Wersja bez `uv`

#### 1. Sklonuj repozytorium i przejdź do katalogu projektu

```bash
git clone git@github.com:UZ-FENS/passthebranch-Kitori2137.git
cd passthebranch-Kitori2137
```
#### 2. Utwórz i aktywuj środowisko wirtualne
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
#### 3. Zainstaluj projekt
```bash
pip install -e .
```
#### 4. Uruchom aplikację
```bash
python -m AOA.app
```
#### 5. Uruchom testy
```bash
pytest
```


---

## Aktualny stan projektu

Na obecnym etapie projekt posiada stabilny fundament techniczny i spełnia założenia pierwszego etapu rozwoju. Aplikacja została uporządkowana architektonicznie, a jej podstawowe moduły działają w sposób spójny i testowalny.

### Zrealizowane elementy

- wydzielenie warstwy `core`,
- oddzielenie logiki aplikacyjnej od warstwy GUI,
- modularna struktura projektu oparta o układ `src/`,
- obsługa generowania danych testowych,
- obsługa wczytywania danych z plików CSV,
- przygotowanie cech do modeli ML,
- trenowanie wielu modeli jednocześnie,
- automatyczny zapis modeli do osobnych plików z unikalnymi nazwami,
- obsługa modeli:
  - Random Forest dla jakości,
  - Gradient Boosting dla opóźnień,
  - Random Forest dla strategii harmonogramowania,
- obsługa heurystycznych modeli STO:
  - `MT`,
  - `MO`,
  - `MZO`,
  - `GENETIC`,
- analiza sumy dodatnich opóźnień dla różnych kolejności zleceń,
- podstawowe operacje analityczne i ewaluacyjne,
- wizualizacje danych i modeli,
- podgląd danych w interfejsie aplikacji,
- dokumentacja użytkownika i dokumentacja teoretyczna,
- testy jednostkowe dla warstwy `core`,
- plik `CHANGELOG.md`,
- przygotowanie projektu do wersjonowania oraz release’ów.

### Aktualny charakter projektu

Projekt jest obecnie działającą aplikacją analityczno-edukacyjną, która pozwala:

- trenować modele na danych przykładowych lub własnych,
- generować własne zestawy danych z kontrolą parametrów wejściowych,
- analizować dane w interfejsie graficznym,
- porównywać różne podejścia do harmonogramowania,
- uruchamiać analizy STO dla ręcznie podanych zleceń,
- generować podstawowe wizualizacje,
- przeglądać wyniki regresji i klasyfikacji,
- zapisywać wyniki oraz modele do plików.

Obecna wersja projektu stanowi działającą bazę do dalszej rozbudowy zarówno pod kątem funkcjonalnym, jak i architektonicznym.

## Plan na kolejny update

- rozbudowa liczby dostępnych modeli,
- dodanie bardziej zaawansowanych wykresów,
- optymalizacja kodu,
- rozszerzenie testów jednostkowych,


## Plany na updaty w przyszłym miesiącu
- dalsze oczyszczenie architektury i refaktoryzacja modułów,
- integracja z dodatkowymi źródłami danych lub modułami wspomagania użytkownika dla części teoretycznej,
- przebudowa struktury main oraz readme z aplikacji,
- kompletny overhaul struktury wizualnej, wprowadzenie początkowych dashboard oraz lepsza dla użytkownika strukura,
- więcej modeli oraz ciekawych wizualizacji,
- helpdesk z agentem/AIKA
