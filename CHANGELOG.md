# CHANGELOG

## [0.3.0] - 2026-04-09

### Added

* Dodano rozszerzone testy jednostkowe dla warstwy `core`, obejmujące m.in.:
  * przepływy usługowe w `core/services.py`,
  * walidację konfiguracji generowania danych,
  * zapis i odczyt paczek modeli,
  * dodatkowe przypadki brzegowe dla `features`, `models`, `data_generation`, `scheduling`, `visualization_service` i `sto_models`.
* Dodano obsługę zapisywania konfiguracji modeli STO jako osobnych paczek modeli, analogicznie do modeli ML.
* Dodano możliwość późniejszego rozwiązywania danych z pliku CSV na podstawie wcześniej zapisanego modelu STO.
* Dodano generowanie osobnych plików wynikowych dla każdego uruchomionego modelu STO.
* Dodano zapis najlepszego wyniku STO do osobnego pliku wynikowego.
* Dodano kolumny wynikowe STO do plików CSV, w tym:
  * `sto_model`,
  * `sto_order`,
  * `sto_start`,
  * `sto_end`,
  * `sto_lateness`,
  * `sto_tardiness_positive`,
  * `sto_cumulative`,
  * `sto_total_for_model`.
* Dodano czytelniejsze logowanie przebiegu analiz STO oraz informacji o zapisanych plikach wynikowych.
* Dodano rozróżnienie paczek modeli według typu (`ml` / `sto`), aby zachować spójny workflow rozwiązywania.

### Changed

* Ujednolicono workflow pracy z modelami STO tak, aby działał podobnie do modeli ML:
  * najpierw zapis modelu,
  * później wybór modelu,
  * następnie wybór danych wejściowych,
  * na końcu zapis wyniku.
* Zmieniono nazewnictwo plików wynikowych tak, aby najpierw wskazywały model, potem dane źródłowe i datę zapisu, np.:
  * `wynik_priority_genetic_nazwa_danych_YYYYMMDD_HHMMSS.csv`,
  * `wynik_priority_best_genetic_nazwa_danych_YYYYMMDD_HHMMSS.csv`.
* Rozszerzono `core/services.py`, aby rozdzielał logikę modeli ML i STO w bardziej jawny i przewidywalny sposób.
* Rozbudowano logikę raportowania STO, aby lepiej pokazywać ranking modeli, szczegóły kolejności oraz wynik najlepszego rozwiązania.
* Zmieniono zachowanie GUI w `main_page.py`, aby zapis i późniejsze użycie modeli STO było bardziej intuicyjne.
* Rozszerzono obsługę postępu dla treningu modeli ML, tak aby użytkownik widział bardziej szczegółowy przebieg wykonywania operacji.

### Fixed

* Naprawiono niespójność pomiędzy logiką rozwiązywania modeli ML i STO.
* Naprawiono sytuację, w której wyniki harmonogramowania mogły pojawiać się mimo braku wybranego modelu `Schedule`.
* Poprawiono warunki zapisu wyników tak, aby użytkownik otrzymywał pliki jednoznacznie powiązane z użytym modelem.
* Poprawiono obsługę walidacji danych wejściowych przy rozwiązywaniu danych na podstawie zapisanych modeli STO.
* Poprawiono testy jednostkowe po zmianach w strukturze raportów STO i sposobie generowania wyników.
* Poprawiono zgodność testów z aktualną logiką `services.py`, w tym z dynamicznym budowaniem nazw plików wynikowych.

### Improved

* Zwiększono spójność architektury pomiędzy modelami ML i heurystykami STO.
* Ułatwiono użytkownikowi identyfikację, który plik wynikowy odpowiada konkretnemu modelowi i konkretnemu zestawowi danych.
* Poprawiono czytelność logów operacyjnych w GUI.
* Zwiększono pokrycie testowe kluczowych ścieżek działania aplikacji.
* Ułatwiono dalszy rozwój mechanizmu wersjonowania i publikacji wydań.


## [0.2.2] - 2026-04-02

### Fixed
- Dodano walidację dodatnich wartości dla parametrów liczbowych używanych w generowaniu danych oraz w analizie STO.
- Zablokowano możliwość podawania wartości ujemnych lub zerowych w polach, które powinny przyjmować wyłącznie wartości dodatnie.
- Dodano komunikat walidacyjny dla nieprawidłowych danych wejściowych:
  - „Głuptasie, czemu wpisujesz ujemne rzeczy. Wpisuj dodatnie”.
- Naprawiono problem powodujący wywalanie aplikacji podczas treningu modeli w `main_page.py`.
- Naprawiono problem powodujący wywalanie aplikacji przy opcji **„Rozwiąż istniejące modele”**.
- Naprawiono obsługę wątków w `main_page.py`, tak aby komunikaty GUI były aktualizowane bezpiecznie z użyciem metod wykonywanych w głównym wątku.
- Naprawiono problem z `DecisionTree` w `visual_page`, wynikający z nieprawidłowego przygotowania figury do renderowania.
- Naprawiono błąd importu w `sto_models.py`, powodujący circular import i uniemożliwiający uruchomienie aplikacji.

### Added
- Dodano centralne parsowanie i walidację konfiguracji generowania danych w warstwie `core/services.py`.
- Dodano funkcję `generate_and_store_datasets_from_config()` do obsługi konfiguracji przekazywanej z GUI.
- Dodano bezpieczne pomocnicze metody do obsługi logów i komunikatów błędów wywoływanych z worker threadów w `main_page.py`.

### Changed
- Przeniesiono dodatkową walidację pól wejściowych z GUI do warstwy `core`, aby zachować brak logiki biznesowej w `main_page.py`.
- Uporządkowano obsługę błędów w `main_page.py`, tak aby spodziewane błędy danych były oddzielone od błędów nieoczekiwanych.
- Usprawniono przepływ generowania danych, treningu i rozwiązywania modeli z zachowaniem architektury opartej na `core/services.py`.
- Usprawniono analizę STO tak, aby błędne dane wejściowe były walidowane spójnie z resztą aplikacji.

### Improved
- Zwiększono stabilność działania głównej zakładki aplikacji.
- Poprawiono odporność aplikacji na błędy użytkownika przy wpisywaniu danych liczbowych.
- Poprawiono kompatybilność modułów analitycznych i wizualizacyjnych po rozbudowie projektu o modele STO.

## [0.2.1] - 2026-03-31

### Fixed
- Naprawiono obsługę błędów w `main_page.py`, tak aby błędy danych były odróżniane od błędów nieoczekiwanych.
- Ograniczono użycie ogólnego `except Exception as e` jako jedynej formy obsługi błędów.
- Dodano zapisywanie pełnego tracebacka do plików logów dla nieoczekiwanych wyjątków.
- Poprawiono diagnostykę problemów podczas:
  - generowania danych,
  - wczytywania plików CSV,
  - treningu modeli,
  - rozwiązywania istniejących modeli,
  - analizy modeli STO.

### Added
- Dodano nowy moduł `utils/error_utils.py`.
- Dodano funkcję zapisu błędów do plików w katalogu `logs/`.
- Dodano kontekst błędu do logów, aby łatwiej identyfikować miejsce awarii.
- Dodano katalog `logs/` do `.gitignore`.

### Changed
- Zmieniono komunikaty błędów wyświetlane użytkownikowi w `main_page.py`.
- Dla błędów spodziewanych, takich jak błędne dane wejściowe, aplikacja pokazuje teraz bardziej precyzyjne komunikaty.
- Dla błędów nieoczekiwanych aplikacja pokazuje ogólny komunikat użytkownikowi, a szczegóły zapisuje do logu.
- Usprawniono logowanie zdarzeń w zakładce głównej, tak aby użytkownik widział informację o zapisaniu szczegółów błędu do pliku.

### Improved
- Zwiększono czytelność i użyteczność komunikatów błędów.
- Ułatwiono debugowanie i dalsze rozwijanie aplikacji.
- Poprawiono odporność interfejsu GUI na trudniejsze do przewidzenia błędy wykonania.
- Poprawiono README o usunięcie z uv "py -m"

## [0.2.0] - 2026-03-30

### Added
- Dodano obsługę heurystycznych modeli STO w osobnym module `core/sto_models.py`.
- Dodano modele STO:
  - `MT` – sortowanie według terminu realizacji,
  - `MO` – sortowanie według najkrótszego czasu wykonania,
  - `MZO` – sortowanie według najdłuższego czasu wykonania,
  - `GENETIC` – model genetyczny / optymalizacyjny minimalizujący STO.
- Dodano raportowanie wyników STO, w tym:
  - kolejność zleceń dla każdego modelu,
  - wartość STO,
  - przebieg liczenia krok po kroku,
  - wskazanie najlepszego i najgorszego modelu.
- Dodano nowe testy jednostkowe dla:
  - `dataset_ops`,
  - `services`,
  - `sto_models`,
  - `visualization_service`.

### Changed
- Przebudowano `main_page.py` i rozszerzono interfejs o:
  - wybór wielu modeli ML jednocześnie,
  - konfigurację liczby rekordów,
  - konfigurację liczby maszyn,
  - konfigurację zakresu czasu produkcji,
  - konfigurację zakresu bufora terminu,
  - wybór kształtów i materiałów,
  - panel podsumowania aktualnych ustawień,
  - panel statusu danych,
  - podgląd danych w zakładce głównej,
  - sekcję analizy STO wraz z polem wynikowym.
- Zmieniono generowanie danych tak, aby użytkownik mógł samodzielnie określać parametry wejściowe.
- Zmieniono logikę terminów z `termin_dni` na `termin_h`.
- Zmieniono przygotowanie danych tak, aby `termin_h` był zawsze większy od `czas_produkcji_h`.
- Zmieniono sposób zapisu modeli:
  - modele nie są już zapisywane wyłącznie jako `model.pkl`,
  - nazwa pliku modelu zawiera wybrane modele, parametry generowania, wybrane kształty, wybrane materiały oraz znacznik czasu.
- Zaktualizowano `guide.md` o:
  - nowy sposób generowania danych,
  - opis modeli STO,
  - opis nowych parametrów treningu i zapisu modeli.
- Zaktualizowano `README.md`:
  - poprawiono instrukcję uruchamiania projektu,
  - dodano poprawny sposób uruchamiania przez `uv`,
  - uproszczono opis instalacji bez `uv`,
  - zaktualizowano strukturę repozytorium,
  - zaktualizowano opis aktualnego stanu projektu.

### Fixed
- Naprawiono układ głównej zakładki tak, aby wszystkie sekcje były dostępne i czytelne.
- Przywrócono możliwość podglądu wygenerowanych lub wczytanych danych w zakładce głównej.
- Poprawiono organizację GUI tak, aby warstwa prezentacji korzystała z gotowych funkcji z `core`, bez umieszczania tam logiki biznesowej.
- Poprawiono strukturę projektu i dokumentacji pod kątem uruchamiania bez ręcznego ustawiania `PYTHONPATH`.

### Improved
- Uporządkowano podział odpowiedzialności między `core` i `gui`.
- Rozszerzono architekturę projektu o osobny moduł dla modeli STO.
- Zwiększono elastyczność eksperymentowania z danymi, modelami i porównywaniem różnych podejść do harmonogramowania.
- Poprawiono przygotowanie projektu do dalszej rozbudowy i kolejnych wersji.

## [0.1.1] - 2026-03-23

### Fixed
- Usunięto logikę biznesową z warstwy GUI w kluczowych modułach aplikacji.
- Przeniesiono operacje generowania i zapisu danych z `gui/pages/main_page.py` do warstwy `core/services.py`.
- Przeniesiono przygotowanie danych do wizualizacji oraz budowę wykresów do `core/visualization_service.py` i modułów pomocniczych w `core/diagrams/`.
- Ograniczono rolę `visual_page.py` do pobrania parametrów od użytkownika i wyświetlenia gotowej figury.
- Przeniesiono analizę danych i przygotowanie wyników z `results_page.py` do funkcji serwisowych w warstwie `core`.
- Ograniczono `results_page.py` do zbierania wyborów użytkownika i prezentacji gotowego wyniku.
- Uporządkowano przepływ ładowania danych tak, aby GUI korzystało wyłącznie z gotowych funkcji udostępnionych przez warstwę `core`.

### Changed
- Zaktualizowano `README.md` i uporządkowano jego strukturę.
- Dodano nowy układ dokumentacji w `README.md`:
  - opis projektu,
  - aktualna struktura repozytorium,
  - instrukcja uruchomienia projektu z `uv`,
  - instrukcja uruchomienia projektu bez `uv`,
  - aktualny stan projektu,
  - plany na kolejny update.
- Doprecyzowano opis architektury projektu oraz aktualny zakres funkcjonalności.
- Uproszczono tryby treningu w GUI przez usunięcie opcji **„Oba”** z wyboru modeli w zakładce głównej.
- Zmieniono interfejs tak, aby użytkownik wybierał wyłącznie pojedynczy tryb treningu: `Quality`, `Delay` albo `Schedule`.

### Improved
- Poprawiono czytelność i spójność architektury projektu.
- Wzmocniono zgodność projektu z wymaganiem oddzielenia logiki aplikacyjnej od warstwy prezentacji.
- Ułatwiono dalszy rozwój aplikacji przez lepszy podział odpowiedzialności między `core` i `gui`.
- Uporządkowano dokumentację projektu, aby była bardziej czytelna i gotowa do prezentacji lub dalszego rozwijania.

## [0.1.0] - 2026-03-20
### Added
- Wyodrębniono warstwę core.
- Dodano moduły data_generation, features, scheduling, models, data_io.
- Dodano testy jednostkowe dla core.
- Rozdzielono GUI od logiki biznesowej.

### Fixed
- Uporządkowano przepływ generowania i podziału danych.
- Ograniczono logikę wykonywaną bezpośrednio w GUI.
