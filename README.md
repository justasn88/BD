# Evoliuciniai skaičiavimai ir jų taikymai

## Apie projektą
Šioje repozitorijoje pateikiami algoritmai, programinis kodas.

## Repozitorijos failų struktūra

*   **`main.cpp`** – **Hibridinio (CS+GA) algoritmo** realizacija (CPU) versija.
*   **`main.py`** – **Memetinio algoritmo (MA)**  ir **Genetinio algoritmo (GA)** realizacija. 
*   **`algo.cu`** – – **Hibridinio (CS+GA) algoritmo** realizacija (GPU) versija.

## Pagrindiniai realizuoti algoritmai
*   **Genetinis algoritmas (GA):** Realizuotas su adaptyvia mutacijos tikimybe, užtikrinančia populiacijos stabilumą skirtingose dimensijose.
*   **Memetinis algoritmas (MA):** Sujungia globalios GA ir lokalia paieška ,,hill climbing" metodu.
*   **Hibridinis CS+GA algoritmas:** Inovatyvus dviejų pakopų modelis. Globaliai paieškai ir ištrūkimui iš nulinio gradiento zonų naudojamas CS, o aptikus gradientą – procesas perjungiamas į GA.

## Tyrimų aplinka
*   **Lokali įranga:** AMD Ryzen 7 6800H, 16GB DDR5 RAM, NVIDIA GeForce RTX 3050 (4GB GDDR6).
