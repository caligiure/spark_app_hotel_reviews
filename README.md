# Progetto Didattico PySpark - Hotel Reviews

Questo progetto mostra come utilizzare PySpark per analizzare un dataset di recensioni di hotel (`Hotel_Reviews.csv`).

Il presente progetto è stato realizzato lavorando sul dataset ``Hotel Reviews'', contenente oltre 515.000 recensioni di alberghi di lusso europei. L'applicativo realizzato consente di effettuare interrogazioni aggregate sul dataset, con l'obiettivo di estrarre insight significativi dai dati.
Sfruttando le potenzialità di elaborazione distribuita offerte dal framework Spark, sono stati realizzati diversi moduli di analisi che consentono di rilevare aspetti temporali, testuali, geospaziali e comportamentali nelle recensioni del dataset. Gli obiettivi principali raggiunti includono:
*   Identificazione dei trend di gradimento degli hotel nel tempo.
*   Analisi dell'influenza di specifici tag (caratteristiche del soggiorno) sul punteggio finale.
*   Segmentazione geografica e analisi della competitività locale.
*   Profilazione degli hotel tramite algoritmi di Machine Learning (Clustering).
*   Studio delle preferenze in base alla nazionalità e tipologia di viaggiatore.

## Analisi del dataset

Il dataset utilizzato `e Hotel Reviews.csv (reperibile su: https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe. 
Questo archivio contiene oltre 515.000 recensioni di hotel di lusso in Europa, raccolte dal sito Booking.com, dove sono pubblicamente accessibili. Ogni riga del dataset corrisponde ad una recensione e presenta 17 campi che
descrivono sia le caratteristiche dell’hotel, sia l’esperienza del cliente.
*   Dimensione File: Circa 238 MB
*   Numero di Righe: 515.738

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
4.  **Winutils e Hadoop** (Necessario su Windows)

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

## Logica di Funzionamento

1.  L’utente avvia l’applicazione tramite script batch (RUN APP Hotel Reviews.bat) o comando Streamlit (python -m streamlit run app.py).
2.  L’app si avvia automaticamente in una finestra del browser predefinito del dispositivo in
uso, ma è anche raggiungibile da altri dispositivi (pc, tablet, smartphone) collegati sulla
stessa rete locale, tramite gli indirizzi specificati nel terminale.
3.  Dopo il caricamento iniziale del dataset in memoria (DataFrame Spark), tramite una
sidebar laterale è possibile selezionare una delle query disponibili.
4.  Ogni query espone parametri specifici (es. numero minimo di recensioni, raggio in km)
modificabili tramite slider o input box. L’esecuzione della query avviene on-demand sfruttando il motore Spark. I risultati vengono visualizzati in-app tramite grafici, tabelle e
mappe interattive.

### Architettura Frontend/Backend

L’applicazione è stata realizzata seguendo una logica Frontend/Backend:
*   Backend (Spark): Il file queries.py contiene la logica di lavoro. Ogni funzione imple-
menta una diversa analisi dei dati, ma tutte rispettano il seguente schema: accetta un
DataFrame Spark in input e restituisce un DataFrame Spark trasformato con i risultati.
*   Frontend (Streamlit): Il file app.py gestisce l’interfaccia utente. All’avvio inizializza
una SparkSession (cachata per efficienza) e carica il dataset. Quando l’utente seleziona
un’analisi, il frontend invoca la funzione corrispondente dal backend, converte i risultati
aggregati (di dimensioni ridotte) in Pandas DataFrame e li visualizza tramite grafici e
tabelle.
   
# Elenco delle query

TO-DO
