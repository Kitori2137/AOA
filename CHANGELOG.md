# CHANGELOG

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
