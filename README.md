# Animals-10 Image Classification Project
Projekat za klasifikaciju slika iz Animals-10 dataset-a, preuzetog sa kaggla, koricenjem konvolucionih neuronskih mreza (CNN).

## Pregled
Projekat implementira  kompletan pipeline za:

- **Analiza podataka** - Ucitavanje, ciscenje, validacija, vizuelizacija
- **Treniranje modela** - CNN arhitektura sa cross-validacijom
- **Poredjenje modela** - 5 razlicitih konfiguracija

  **Dataset: Animals-10**
  - 26.178 ukupno slika
  - 10 klasa: leptir, macka, kokoska, krava, pas, slon, konj, ovca, pauk, veverica

  ## Notebook-ovi
  **Analiza.ipynb**
  Ovaj notebook služi za kompletnu eksploratornu analizu podataka (EDA) Animals-10 skupa podataka.
  Koristi se zajedno sa dataset.py koji predstavlja klasu za zivotinje.

  Sekcije:
- Import biblioteka
- Učitavanje podataka
- Validacija
- Provera nedostajucih podataka
- Ciscenje
- Deskriptivna statistika
- Vizualizacija distribucije klasa
- Provera balansiranosti
- Analiza svojstava slika
- Vizualizacija uzoraka
- Analiza distribucije piksela
- Korelaciona analiza
- Čuvanje obrađenih podataka

**Treniranje.ipynb**
Ovaj notebook sluzi za definisanje arhitekture,treniranje i poredjenje modela.

Sekcije: 

- Import biblioteka
- Pregled arhitekture
- Učitavanje podataka
- Vizualizacija uzoraka
- Definisanje 5 konfiguracija
- Definisanje hiperparametara
- Ucitavanje podataka
- Vizualizacija poredjenja
- Learning curves
- Confusion matrix
- Cuvanje rezultata

## Aplikacija - Pokretanje
Aplikacija se pokrece tako sto se pokrene fajl Aplikacija.bat koji ce kreirati docker kontejner sa svim potrebnim zavisnostima. Aplikacija ce nakon pokretanja biti dostupna na url-u http://localhost:8000/.

## Aplikacija - Uputstvo za korišćenje

**Dodavanje slike**
Slika se moze dodati na dva nacina:

- klikom na polje za upload i odabirom slike sa racunara
- prevlacenjem slike (drag & drop) u predvidjeno polje

**Pokretanje klasifikacije**

Klikom na dugme "Predvidi klasu" model obradjuje sliku.

<img width="2554" height="706" alt="a1" src="https://github.com/user-attachments/assets/6e8c53de-be4e-4d3f-94f6-6558dea35ab5" />

**Prikaz rezultata**

Nakon sto model obradi sliku ispod nje ce se pojaviti top 3 klase koje je model predvideo.

<img width="2559" height="933" alt="a2" src="https://github.com/user-attachments/assets/079601e0-f67d-4534-bc4e-e69a9c447ca4" />

