# AOA – Aplikacja Optymalnego Algorytmowania

## Opis projektu

AOA to rozwijana aplikacja napisana w Pythonie, której celem jest wspomaganie analizy danych produkcyjnych, trenowania modeli uczenia maszynowego, porównywania metod harmonogramowania oraz prezentacji wyników w formie tabelarycznej i wizualnej.

Projekt został przygotowany w architekturze modułowej, z wyraźnym podziałem na niezależne warstwy odpowiedzialności:

- warstwę `core`, odpowiadającą za logikę aplikacji, generowanie i przetwarzanie danych, trenowanie modeli, ewaluację wyników, obsługę harmonogramowania oraz przygotowanie danych do wizualizacji,
- warstwę `gui`, odpowiedzialną za graficzny interfejs użytkownika aplikacji desktopowej,
- warstwę `cli`, umożliwiającą pełną obsługę aplikacji z poziomu terminala, bez uruchamiania interfejsu graficznego,
- katalog `tests`, zawierający testy jednostkowe i testy przepływów dla warstwy `core` oraz interfejsu CLI,
- katalog `docs`, zawierający dokumentację użytkową i teoretyczną projektu,
- katalog `logs`, przechowujący logi błędów i informacje diagnostyczne pomocne podczas debugowania.

Aplikacja umożliwia między innymi:

- generowanie przykładowych danych produkcyjnych,
- wczytywanie danych z plików CSV,
- trenowanie modeli ML dla jakości, opóźnień i strategii harmonogramowania,
- korzystanie zarówno z klasycznych modeli opartych o `scikit-learn`, jak i z eksperymentalnego backendu `TabPFN`,
- analizę wyników regresji i klasyfikacji,
- uruchamianie metod harmonogramowania STO i porównywanie ich wyników,
- tworzenie wykresów i wizualizacji danych,
- zapis modeli, raportów i wyników do plików,
- obsługę pełnych przepływów pracy zarówno z poziomu GUI, jak i z terminala.

Projekt ma charakter rozwojowy i stanowi bazę pod dalszą rozbudowę o nowe algorytmy, dokładniejsze analizy, kolejne metody optymalizacji, bardziej zaawansowane moduły wspomagania decyzji oraz dalsze rozszerzanie obsługi terminalowej i automatyzacji pracy.

## Aktualna struktura repozytorium

```text
TOOLS/
├── docs/
│   ├── guide.md
│   └── theory.md
├── data/
├── logs/
├── models/
├── src/
│   └── AOA/
│       ├── __init__.py
│       ├── app.py
│       ├── cli.py
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
│       │   ├── tabpfn_models.py
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
│           ├── error_utils.py
│           ├── logging_utils.py
│           └── threading_utils.py
├── tests/
│   ├── cli/
│   │   ├── test_cli_commands.py
│   │   ├── test_cli_interactive.py
│   │   ├── test_cli_main.py
│   │   └── test_cli_workflow.py
│   └── core/
│       ├── test_data_generation.py
│       ├── test_data_generation_extended.py
│       ├── test_data_io.py
│       ├── test_dataset_ops.py
│       ├── test_evaluation.py
│       ├── test_features.py
│       ├── test_features_extended.py
│       ├── test_io_and_split_extended.py
│       ├── test_model_pack_flows.py
│       ├── test_models.py
│       ├── test_models_extended.py
│       ├── test_scheduling.py
│       ├── test_scheduling_extended.py
│       ├── test_services_extra.py
│       ├── test_services_flows.py
│       ├── test_services_extended.py
│       ├── test_sto_models.py
│       ├── test_sto_models_extended.py
│       ├── test_tabpfn_models.py
│       ├── test_visualization_service.py
│       └── test_visualization_service_extended.py
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
uv sync --dev
uv run aoa-cli
```

#### 3. Uruchom testy
```bash
uv run pytest
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

Na obecnym etapie projekt posiada stabilny fundament techniczny i obejmuje już nie tylko aplikację desktopową, ale również równoległy interfejs terminalowy. Aplikacja została uporządkowana architektonicznie, a jej główne moduły działają w sposób spójny, testowalny i możliwy do rozwijania w kolejnych wersjach.

### Zrealizowane elementy

- wydzielenie warstwy `core`,
- oddzielenie logiki aplikacyjnej od warstwy GUI,
- dodanie warstwy `cli`, umożliwiającej obsługę aplikacji z poziomu terminala,
- modularna struktura projektu oparta o układ `src/`,
- obsługa generowania danych testowych,
- obsługa wczytywania danych z plików CSV,
- przygotowanie cech do modeli ML,
- trenowanie wielu modeli jednocześnie,
- automatyczny zapis modeli do osobnych plików z unikalnymi nazwami,
- obsługa modeli klasycznych:
  - Random Forest dla jakości,
  - Gradient Boosting dla opóźnień,
  - Random Forest dla strategii harmonogramowania,
- dodanie eksperymentalnego backendu `TabPFN` dla wybranych modeli ML,
- obsługa heurystycznych modeli STO:
  - `MT`,
  - `MO`,
  - `MZO`,
  - `GENETIC`,
- analiza sumy dodatnich opóźnień dla różnych kolejności zleceń,
- podstawowe operacje analityczne i ewaluacyjne,
- wizualizacje danych i modeli,
- podgląd danych w interfejsie aplikacji,
- obsługa pełnych przepływów pracy zarówno w GUI, jak i w CLI,
- obsługa komend terminalowych do:
  - generowania danych,
  - treningu modeli,
  - rozwiązywania zapisanym modelem,
  - uruchamiania analiz STO,
  - pracy w trybie `workflow`,
  - pracy w trybie `interactive`,
- dokumentacja użytkownika i dokumentacja teoretyczna,
- testy jednostkowe i testy przepływów dla warstwy `core`,
- dodanie testów dla warstwy `cli`,
- logowanie błędów do katalogu `logs`,
- plik `CHANGELOG.md`,
- przygotowanie projektu do wersjonowania oraz release’ów.

### Aktualny charakter projektu

Projekt jest obecnie działającą aplikacją analityczno-edukacyjną, która pozwala:

- trenować modele na danych przykładowych lub własnych,
- generować własne zestawy danych z kontrolą parametrów wejściowych,
- analizować dane w interfejsie graficznym,
- obsługiwać najważniejsze funkcje również z poziomu terminala,
- porównywać różne podejścia do harmonogramowania,
- uruchamiać analizy STO dla ręcznie podanych zleceń,
- korzystać zarówno z klasycznych modeli `scikit-learn`, jak i z backendu `TabPFN`,
- generować podstawowe wizualizacje,
- przeglądać wyniki regresji i klasyfikacji,
- zapisywać wyniki, raporty oraz modele do plików,
- wykonywać pełne scenariusze pracy bez uruchamiania GUI.

Obecna wersja projektu stanowi działającą bazę do dalszej rozbudowy zarówno pod kątem funkcjonalnym, jak i architektonicznym. Projekt rozwija się już nie tylko jako aplikacja desktopowa, ale również jako narzędzie możliwe do wykorzystania w pracy terminalowej, testach automatycznych i dalszej automatyzacji procesów analitycznych.

## Plan na kolejny update

- rozbudowa liczby dostępnych modeli,
- dodanie bardziej zaawansowanych wykresów / rozwinięcie istniejących,
- optymalizacja kodu,
- dashboardy finalnie,


## Plany na updaty 
- integracja z dodatkowymi źródłami danych lub modułami wspomagania użytkownika dla części teoretycznej,
- przebudowa struktury main oraz readme z aplikacji,
- kompletny overhaul struktury wizualnej, wprowadzenie początkowych dashboard oraz lepsza dla użytkownika strukura,
- więcej modeli oraz ciekawych wizualizacji,
- helpdesk z agentem/AIKA
