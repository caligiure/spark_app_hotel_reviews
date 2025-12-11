# Progetto Didattico PySpark - Hotel Reviews

Questo progetto mostra come utilizzare PySpark per analizzare un dataset di recensioni di hotel (`Hotel_Reviews.csv`).

## Prerequisiti

1.  **Python** (installato)
2.  **Java JDK** (installato)
3.  **PySpark** (installato via `pip install pyspark`)
4.  **Winutils** (Necessario su Windows)

### Configurazione Winutils (IMPORTANTE)

Per eseguire Spark su Windows senza errori, è necessario configurare `winutils.exe` e `hadoop.dll`.

1.  **Scarica i binari**:
    *   Vai su un repository affidabile come [cdarlint/winutils](https://github.com/cdarlint/winutils).
    *   Naviga nella cartella corrispondente alla tua versione di Hadoop (solitamente Spark include Hadoop 3.x, quindi prova la cartella `hadoop-3.2.2` o `hadoop-3.3.5`).
    *   Scarica `winutils.exe` e `hadoop.dll` dalla cartella `bin`.

2.  **Crea la cartella**:
    *   Crea una cartella nel tuo disco, ad esempio `C:\hadoop\bin`.
    *   Copia i file scaricati (`winutils.exe` e `hadoop.dll`) dentro `C:\hadoop\bin`.

3.  **Imposta le Variabili d'Ambiente**:
    *   Apri "Modifica le variabili di ambiente relative al sistema" (cerca "env" nel menu Start).
    *   Crea una nuova variabile utente o di sistema:
        *   Nome: `HADOOP_HOME`
        *   Valore: `C:\hadoop` (senza `\bin`)
    *   Modifica la variabile `Path`:
        *   Aggiungi `%HADOOP_HOME%\bin`.

## Esecuzione

1.  Apri il terminale nella cartella del progetto.
2.  Esegui lo script:

```bash
python main.py
```

## Cosa fa lo script

1.  Inizializza una `SparkSession`.
2.  Carica il file `Hotel_Reviews.csv`.
3.  Calcola la media del `Reviewer_Score` per ogni `Hotel_Name`.
4.  Filtra gli hotel con meno di 20 recensioni.
5.  Mostra i primi 10 hotel per punteggio medio.
