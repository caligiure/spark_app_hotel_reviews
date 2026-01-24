# Progetto Didattico PySpark - Hotel Reviews

Questo progetto mostra come utilizzare PySpark per analizzare un dataset di recensioni di hotel (`Hotel_Reviews.csv`).

## Analisi del dataset

### Campi del dataset

* Hotel_Address: indirizzo dell'hotel
* Additional_Number_of_Scoring: numero di valutazioni aggiuntive
* Review_Date: data della recensione
* Average_Score: punteggio medio calcolato su tutte le recensioni ricevute dall'hotel nell'ultimo anno
* Hotel_Name: nome dell'hotel
* Reviewer_Nationality: nazionalità del recensore
* Negative_Review: recensione negativa (se non presente il campo contiene "No Negative")
* Review_Total_Negative_Word_Counts: numero di parole negative nella recensione
* Total_Number_of_Reviews: numero totale di recensioni dell'hotel
* Positive_Review: recensione positiva (se non presente il campo contiene "No Positive")
* Review_Total_Positive_Word_Counts: numero di parole positive nella recensione
* Total_Number_of_Reviews_Reviewer_Has_Given: numero totale di recensioni che ha dato il recensore
* Reviewer_Score: punteggio che il recensore ha dato all'hotel
* Tags: tag associati alla recensione
* days_since_review: giorni trascorsi fra la pubblicazione e lo scraping della recensione
* lat: latitudine dell'hotel
* lng: longitudine dell'hotel

## Prerequisiti

1.  **Python 3.11** (testato con 3.11.9)
2.  **Java JDK**
3.  **pip install pyspark pandas scikit-learn numpy streamlit altair matplotlib**
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
    *   Apri "Modifica le variabili di ambiente relative al sistema"
    *   Crea una nuova variabile utente o di sistema:
        *   Nome: `HADOOP_HOME`
        *   Valore: `C:\hadoop` (senza `\bin`)
    *   Modifica la variabile `Path`:
        *   Aggiungi `%HADOOP_HOME%\bin`.

## Avvio dell'applicazione

* Esegui il file RUN_APP_Hotel_Reviews.bat

Oppure:

* Esegui il comando: python -m streamlit run app.py

### Cosa fa lo script (app.py) di avvio con interfaccia grafica

TO-DO
   
# Elenco delle query

TO-DO
