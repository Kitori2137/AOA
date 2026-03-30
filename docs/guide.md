# 📊 PRODUCTION OPTIMIZATION – INSTRUKCJA UŻYTKOWNIKA

═════════════════════════════════════════════════════

## 📋 PRZEGLĄD

Aplikacja umożliwia analizę danych produkcyjnych, trenowanie modeli uczenia maszynowego oraz wspomaganie decyzji związanych z planowaniem produkcji.

System wspiera obecnie dwa główne obszary działania:

- modele ML do predykcji jakości, opóźnień i strategii harmonogramowania,
- modele heurystyczne STO do porównywania kolejności realizacji zleceń.

Aplikacja pozwala między innymi na:

- generowanie własnych danych testowych,
- wybór parametrów generowania danych,
- wybór wielu modeli ML do jednoczesnego treningu,
- zapis wielu modeli do osobnych plików,
- analizę kolejności zleceń według różnych metod heurystycznych,
- wizualizację danych i modeli,
- przegląd wyników analitycznych w interfejsie graficznym.

### Obsługiwane modele ML

- **Random Forest** – predykcja jakości,
- **Gradient Boosting** – predykcja opóźnień,
- **Random Forest** – wybór strategii harmonogramowania.

### Obsługiwane modele STO

- **MT** – sortowanie według terminu realizacji,
- **MO** – sortowanie według najkrótszego czasu wykonania,
- **MZO** – sortowanie według najdłuższego czasu wykonania,
- **GENETIC** – model genetyczny / optymalizacyjny minimalizujący STO.

═════════════════════════════════════════════════════

## 🎓 TRYB UCZENIA – trenowanie modeli ML

### 1. Wybór modeli

Użytkownik może zaznaczyć jeden lub wiele modeli jednocześnie:

- **Quality** – przewidywanie jakości produkcji,
- **Delay** – przewidywanie opóźnienia realizacji,
- **Schedule** – przewidywanie strategii harmonogramowania.

W odróżnieniu od wcześniejszych wersji aplikacji można zaznaczyć kilka modeli równocześnie i trenować je podczas jednej operacji.

### 2. Parametry generowania danych

W aplikacji można samodzielnie określić parametry generowanego zbioru danych, między innymi:

- liczbę rekordów,
- liczbę maszyn,
- `test_size`,
- `seed`,
- minimalny i maksymalny czas produkcji,
- minimalny i maksymalny bufor terminu,
- wybór kształtów,
- wybór materiałów.

### 3. Struktura danych wejściowych

Dane wejściowe w formacie CSV powinny zawierać kolumny:

- `cena`
- `odpad`
- `termin_h`
- `czas_produkcji_h`
- `ksztalt`
- `material`
- `x`
- `y`
- `z`

### 4. Ważna zmiana – termin w godzinach

W obecnej wersji aplikacji termin realizacji jest zapisany jako:

- `termin_h`

czyli termin w godzinach, a nie w dniach.

Podczas generowania danych system pilnuje, aby nie powstały terminy niemożliwe do realizacji, co oznacza że:

- `termin_h` jest zawsze większy od `czas_produkcji_h`

Dzięki temu generowany zbiór jest bardziej realistyczny i spójny logicznie.

### 5. Start treningu

Aby rozpocząć trening:

- zaznacz wybrane modele ML,
- wygeneruj dane lub wczytaj plik CSV,
- kliknij przycisk **„Trenuj wybrane modele”**,
- obserwuj komunikaty i postęp w polu logów.

### 6. Zapis modeli

Wytrenowane modele zapisywane są automatycznie do katalogu `models/`.

Każdy model zapisywany jest jako osobny plik z nazwą zawierającą między innymi:

- wybrane modele,
- parametry generowania,
- wybrane kształty,
- wybrane materiały,
- znacznik czasu.

Dzięki temu można później łatwo porównywać wiele różnych eksperymentów i wersji treningu.

═════════════════════════════════════════════════════

## 🧠 TRYB STO – analiza kolejności zleceń

### 1. Cel analizy STO

Moduł STO służy do porównania różnych sposobów ustawienia kolejności zleceń.

Dla każdego wybranego modelu obliczana jest suma dodatnich opóźnień, czyli:

- opóźnienie liczone tylko wtedy, gdy zakończenie zlecenia przekracza jego termin,
- wartości dodatnie są sumowane,
- wynik końcowy określany jest jako **STO**.

Im mniejsze STO, tym lepsza kolejność realizacji zleceń.

### 2. Dane wejściowe

Użytkownik podaje ręcznie:

- listę zleceń, np. `Z1,Z2,Z3`
- czasy wykonania, np. `10,20,100`
- terminy, np. `150,30,110`

Liczba zleceń, czasów i terminów musi być taka sama.

### 3. Modele STO

#### **MT**
Model sortuje zlecenia według terminu realizacji rosnąco.

#### **MO**
Model sortuje zlecenia według najkrótszego czasu wykonania.

#### **MZO**
Model sortuje zlecenia według najdłuższego czasu wykonania.

#### **GENETIC**
Model genetyczny / optymalizacyjny szuka takiej kolejności zleceń, która daje możliwie najmniejsze STO.

### 4. Wyniki analizy STO

Po uruchomieniu analizy aplikacja pokazuje:

- kolejność zleceń dla każdego wybranego modelu,
- wartość STO dla każdego modelu,
- przebieg liczenia krok po kroku,
- informację, który model był najlepszy,
- informację, który model był najgorszy.

Raport STO pojawia się tylko wtedy, gdy zaznaczono przynajmniej jeden model heurystyczny STO.

### 5. Przykład

Dla danych:

- zlecenia: `Z1, Z2, Z3`
- czasy: `10, 20, 100`
- terminy: `150, 30, 110`

możliwe wyniki są następujące:

- **MT** → `Z2, Z3, Z1` → STO = `10`
- **MO** → `Z1, Z2, Z3` → STO = `20`
- **MZO** → `Z3, Z2, Z1` → STO = `90`

Na tej podstawie aplikacja wskaże najlepszą i najgorszą kolejność.

═════════════════════════════════════════════════════

## 🔍 TRYB ROZWIĄZANIA – predykcja i priorytety

### 1. Wybór modelu i danych

Aby wykonać predykcję:

- kliknij **„Rozwiąż istniejące modele”**,
- wskaż zapisany model `.pkl`,
- wybierz plik danych `.csv`.

### 2. Predykcje i priorytety

W zależności od zawartości pliku modelu aplikacja może obliczyć:

- `pred_quality` – przewidywaną jakość,
- `pred_delay` – przewidywane opóźnienie,
- `priority` – wskaźnik priorytetu wyznaczany jako relacja jakości do przewidywanego opóźnienia.

### 3. Wyniki

Wyniki:

- są wyświetlane w aplikacji,
- zapisywane są do pliku `data/wynik_priority.csv`,
- prezentują między innymi TOP 10 zleceń o najwyższym priorytecie.

═════════════════════════════════════════════════════

## 📈 WIZUALIZACJA DANYCH

### 1. Wczytanie pliku

Wybierz plik CSV zawierający dane, które chcesz analizować.

### 2. Wybór parametrów wykresu

Użytkownik może:

- wybrać kolumny dla osi **X** i **Y**,
- określić typ wykresu.

Dostępne typy wykresów:

- Scatter,
- Line,
- Histogram,
- Boxplot,
- Gantt,
- DecisionTree,
- CorrelationMatrix,
- SimilarityMatrix.

### 3. Generowanie wykresu

Aby utworzyć wykres:

- kliknij przycisk **„Rysuj”**,
- wykres zostanie wyświetlony w dolnym panelu aplikacji.

### 4. Zastosowanie wizualizacji

Wizualizacje pomagają analizować:

- zależności pomiędzy cechami,
- rozkłady danych,
- podobieństwa i korelacje,
- przebieg harmonogramu produkcji,
- uproszczoną strukturę modelu decyzyjnego.

═════════════════════════════════════════════════════

## 🧾 PODGLĄD DANYCH I TRANSFORMACJE

### 1. Wczytanie danych

- przejdź do zakładki **Results**,
- kliknij **„Wczytaj plik CSV”**,
- wybierz plik, który chcesz przeanalizować.

### 2. Wybór kolumn

Możesz zaznaczyć:

- wszystkie kolumny,
- tylko wybrane kolumny do analizy.

### 3. Dostępne transformacje danych

#### Surowe
Dane pozostają w oryginalnej postaci.

#### MinMax Normalizacja
Skalowanie wartości do przedziału 0–1.

#### Standaryzacja
Transformacja do średniej 0 i odchylenia standardowego 1.

#### Logarytm
Zmniejszenie wpływu wartości skrajnych przy użyciu funkcji `log1p`.

#### Skalowanie 0–1
Proste przeskalowanie każdej kolumny do przedziału 0–1.

### 4. Analiza danych

Po wybraniu kolumn i transformacji użytkownik może uruchomić:

- analizę regresyjną,
- analizę klasyfikacyjną.

Wyniki oraz wybrane dane są prezentowane w polu tekstowym.

═════════════════════════════════════════════════════

## 🛠️ ROZWIĄZYWANIE PROBLEMÓW

**„Brak danych”**  
Wczytaj lub wygeneruj plik CSV.

**„Nie wybrano kolumny”**  
Zaznacz co najmniej jedną kolumnę do analizy.

**Brak danych treningowych**  
Najpierw wygeneruj dane lub wczytaj plik CSV.

**Brak modelu STO**  
Zaznacz przynajmniej jeden model STO: `MT`, `MO`, `MZO` lub `GENETIC`.

**Niska jakość predykcji**  
Zwiększ liczbę danych lub popraw ich jakość.

**Brak modelu `.pkl`**  
Upewnij się, że w katalogu `models/` znajduje się zapisany model.

═════════════════════════════════════════════════════

## 📁 LOKALIZACJA PLIKÓW

**Modele:**  
pliki `.pkl` w katalogu `models/`

**Dane:**  
pliki CSV w katalogu projektu lub w folderze `data/`

**Wyniki predykcji:**  
`data/wynik_priority.csv`

**Dokumentacja:**  
`docs/guide.md`  
`docs/theory.md`
