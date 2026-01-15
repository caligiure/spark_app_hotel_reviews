# Progetto Didattico PySpark - Hotel Reviews

Questo progetto mostra come utilizzare PySpark per analizzare un dataset di recensioni di hotel (`Hotel_Reviews.csv`).

## Analisi del dataset

### Campi del dataset

Hotel_Address,Additional_Number_of_Scoring,Review_Date,Average_Score,Hotel_Name,Reviewer_Nationality,Negative_Review,Review_Total_Negative_Word_Counts,Total_Number_of_Reviews,Positive_Review,Review_Total_Positive_Word_Counts,Total_Number_of_Reviews_Reviewer_Has_Given,Reviewer_Score,Tags,days_since_review,lat,lng

### Spiegazione dei campi

Hotel_Address: indirizzo dell'hotel
Additional_Number_of_Scoring: numero di valutazioni aggiuntive
Review_Date: data della recensione
Average_Score: punteggio medio
Hotel_Name: nome dell'hotel
Reviewer_Nationality: nazionalità del recensore
Negative_Review: recensione negativa
Review_Total_Negative_Word_Counts: numero di parole negative nella recensione
Total_Number_of_Reviews: numero totale di recensioni
Positive_Review: recensione positiva
Review_Total_Positive_Word_Counts: numero di parole positive nella recensione
Total_Number_of_Reviews_Reviewer_Has_Given: numero totale di recensioni che ha dato il recensore
Reviewer_Score: punteggio del recensore
Tags: tag associati alla recensione
days_since_review: giorni trascorsi dalla recensione
lat: latitudine
lng: longitudine

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

## Esecuzione con interfaccia grafica

1. Avvia run_gui.bat

### Cosa fa lo script (app.py) di avvio con interfaccia grafica

1. TO-DO

## Alternativa: esecuzione da terminale (solo per test)

1.  Apri il terminale nella cartella del progetto.
2.  Esegui lo script:
        python main.py

### Cosa fa lo script (main.py) del test da terminale

1.  Inizializza una `SparkSession`.
2.  Carica il file `Hotel_Reviews.csv`.
3.  Calcola la media del `Reviewer_Score` per ogni `Hotel_Name`.
4.  Filtra gli hotel con meno di 20 recensioni.
5.  Mostra i primi 10 hotel per punteggio medio.

   
# Elenco delle query

## 1. get_top_hotels (main.py)

## 2. Estimate future customer satisfaction (query di predizione, ml_model.py)
Questa query ha l'obiettivo di **predire il futuro grado di soddisfazione dei clienti di ogni hotel**, basandosi sul punteggio medio che ogni hotel ha ottenuto nelle valutazioni e sul numero di recensioni che ha ricevuto.
In particolare, la query predice il **Reviewer_Score** della prossima valutazione di ogni hotel, basandosi sul suo *Average_Score* e sul suo *Total_Number_of_Reviews*.

La query implementa una **Regressione Lineare** grazie all'uso di **Spark MLlib**, realizzando un modello di **machine learning** addestrato sul dataset di recensioni di hotel. Poichè il dataset è statico, non è possibile addestrare il modello in tempo reale, ma è possibile addestrare il modello una volta e poi utilizzarlo per fare predizioni in tempo reale.

L'addestramento del modello avviene nel file `ml_model.py`, con la seguente logica:
1. Seleziona le colonne `Average_Score`, `Total_Number_of_Reviews` e `Reviewer_Score` (rimuovendo le righe con valori nulli).
2. Combina le colonne selezionate in una singola colonna di input per il modello, chiamata `features`.
3. Divide i dati in training e test set (80% per training, 20% per test).
4. Inizializza un modello di Regressione Lineare e lo addestra sul training set.
5. TO-DO: spiega come funziona la pipeline e come funziona il modello di Regressione Lineare.
6. TO-DO: spiega come funziona predictions e come funziona evaluator.
7. TO-DO: spiega cosa sono RMSE e R2.
8. TO-DO: spiega cosa sono coefficients e intercept.
9. Spiega cosa restiuisce il modello addestrato.

Output:
Model Trained. RMSE: 1.5239, R2: 0.1335                                                      
Coefficients: [1.083499728270512,-5.373252253753084e-06]
Intercept: -0.687615263580646

Interpretazione dei risultati:
1. L'errore medio (RMSE = 1.52):
    * Questo è l'indicatore più diretto della precisione: se il modello prevede che un utente darà voto 9.0, il voto reale sarà probabilmente compreso tra 7.5 e 10. (cioè ha un margine di errore medio di circa 1.5 punti).
    * Su una scala di voti da 1 a 10, sbagliare di 1.5 punti è un errore significativo, la stima è molto "larga".
2. La capacità di spiegazione (R2 = 0.13):
    * Questo valore (13%) ci dice che il modello capisce solo una piccola parte del comportamento dei clienti.
    * Significa che l'87% del voto di un cliente dipende da fattori che il modello non conosce (es. pulizia della camera specifica, cortesia dello staff in quel momento, umore del cliente), e non dalla **media storica dell'hotel**.

In sintesi: Il modello può dare una stima del futuro grado di soddisfazione dei clienti di ogni hotel, ma questa stima sarà quasi sempre molto vicina alla media storica dell'hotel.