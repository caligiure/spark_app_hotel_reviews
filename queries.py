from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, DateType
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def get_top_hotels_by_nation(df, n=10):
    """
    Raggruppa gli hotel per nazione e trova gli n migliori per ogni nazione
    basandosi su un criterio di punteggio (default: Average_Score).
    Args:
        df: DataFrame PySpark con i dati degli hotel
        n: Numero di hotel da mostrare per ogni nazione
    Returns:
        DataFrame PySpark con le colonne selezionate e i top n hotel per nazione
    """
    # 1. Deduplicazione per Hotel: Conserviamo una sola riga per hotel
    # Nota: Hotel_Address, Average_Score e Total_Number_of_Reviews sono costanti per lo stesso Hotel_Name
    unique_hotels_df = df.select("Hotel_Name", "Hotel_Address", "Average_Score", "Total_Number_of_Reviews") \
                         .dropDuplicates(["Hotel_Name"])

    # 2. Estrazione della Nazione (la nazione è l'ultima parola dell'indirizzo)
    df_with_nation = unique_hotels_df.withColumn("Nation_Raw", F.element_at(F.split(F.col("Hotel_Address"), " "), -1))
    # Nota: uso le funzioni native di Spark (F.split, F.element_at, etc.) perchè è molto più efficiente 
    # rispetto all'uso delle funzioni built-in di Python (le funzioni di spark vengono eseguite direttamente nella JVM)
    
    # Correggiamo le nazioni multi-parole
    df_with_nation = df_with_nation.withColumn(
        "Nation", 
        F.when(F.col("Nation_Raw") == "Kingdom", "United Kingdom")
         .otherwise(F.col("Nation_Raw"))
    )
    
    # 3. Ranking (Top N per Nazione)
    # Definiamo una finestra partizionata per Nazione e ordinata per Average_Score decrescente
    # Se due hotel hanno lo stesso punteggio, vince chi ha più recensioni (Total_Number_of_Reviews).
    window_spec = Window.partitionBy("Nation").orderBy(F.col("Average_Score").desc(), F.col("Total_Number_of_Reviews").desc())
    # Nota: una Window Function permette di eseguire calcoli su un gruppo di righe correlate alla riga corrente, 
    # senza collassare le righe in una sola (come fa invece groupBy).

    # Aggiungiamo il rank (F.rank() assegna un numero unico a ogni riga all'interno di ogni partizione)
    df_ranked = df_with_nation.withColumn("rank", F.rank().over(window_spec))
    
    # 4. Filtriamo i top n
    top_hotels = df_ranked.filter(F.col("rank") <= n)
    
    # Selezioniamo e ordiniamo per una visualizzazione pulita
    result = top_hotels.select(
        "Nation",
        "Hotel_Name",
        "Average_Score",
        "Total_Number_of_Reviews",
        "Hotel_Address"
    ).orderBy("Nation", F.col("Average_Score").desc())
    # Nota: orderBy() impone questo ordinamento: 
    # 1. per Nation (in ordine alfabetico crescente)
    # 2. per Average_Score (in ordine decrescente)
    return result

def analyze_review_trends(df, min_number_of_reviews = 30):
    """
    Analizza il trend temporale dei punteggi delle recensioni per ogni hotel.
    Utilizza una Pandas UDF per calcolare la regressione lineare su ogni gruppo.
    Args:
        df: DataFrame PySpark con i dati degli hotel
        min_number_of_reviews: Numero minimo di recensioni per hotel
    Returns:
        DataFrame PySpark con le colonne selezionate e i top n hotel per nazione
    """

    # UDF Pandas che verrà eseguita su ogni gruppo (Hotel)
    # pdf è un pandas DataFrame contenente le recensioni di UN solo hotel
    # Nota: è importante definire una funzione Pandas, che sarà eseguita con applyInPandas,
    # in modo tale che Spark possa mantenere il parallelismo e non dover eseguire la UDF in un unico worker.
    def calculate_trend(pdf):
        if pdf.empty or len(pdf) < min_number_of_reviews: # Non abbastanza dati per un trend, restituisce un DataFrame vuoto
            return pd.DataFrame()
            
        # 1. Ordina per data (fondamentale per il trend temporale) ed elimina le righe con data non valida
        # coerce: converte le date non valide in NaN
        pdf['Review_Date'] = pd.to_datetime(pdf['Review_Date'], format='%m/%d/%Y', errors='coerce')
        pdf = pdf.dropna(subset=['Review_Date'])
        pdf = pdf.sort_values('Review_Date')
        if len(pdf) < min_number_of_reviews: # Non abbastanza dati per un trend, restituisce un DataFrame vuoto
            return pd.DataFrame()

        # 2. Prepara dati per regressione
        # X: Tempo convertito in numeri ordinali
        # y: Punteggio recensione
        pdf['Date_Ordinal'] = pdf['Review_Date'].map(pd.Timestamp.toordinal)
        X = pdf['Date_Ordinal'].values.reshape(-1, 1) # Reshape per LinearRegression (converte in array di array)
        y = pdf['Reviewer_Score'].values
        
        # 3. Fit Modello Lineare per correlazione tra data e punteggio
        model = LinearRegression() # da sklearn.linear_model
        model.fit(X, y)         # Calcola il modello lineare: 
                                # trova la linea retta (y = mx+q) che meglio si adatta ai dati (best fit line)
                                # dove m è il coefficiente angolare e q è l'intercetta
        slope = model.coef_[0]  # In questo caso ci interessa solo il valore di m (slope), che è il nostro trend temporale
        # Se slope > 0: Trend Crescente
        # Se slope < 0: Trend Decrescente
        
        return pd.DataFrame({
            "Hotel_Name": [pdf['Hotel_Name'].iloc[0]], # prende il nome del hotel dalla prima riga del DataFrame
            "Trend_Slope": [float(slope)],
            "Review_Count": [len(pdf)],
            "Average_Score_Calculated": [float(y.mean())],
            "Average_Score": [round(float(pdf['Average_Score'].iloc[0]), 1)],
            "First_Review_Date": [pdf['Review_Date'].min()],
            "Last_Review_Date": [pdf['Review_Date'].max()]
        })

    # Schema di output della UDF
    schema = StructType([
        StructField("Hotel_Name", StringType(), True),
        StructField("Trend_Slope", FloatType(), True),
        StructField("Review_Count", IntegerType(), True),
        StructField("Average_Score_Calculated", FloatType(), True),
        StructField("Average_Score", FloatType(), True),
        StructField("First_Review_Date", DateType(), True),
        StructField("Last_Review_Date", DateType(), True)
    ])

    # Riduce il dataset su cui lavorare, selezionando solo le colonne strettamente necessarie 
    # e filtrando gli hotel con almeno {min_number_of_reviews} recensioni,
    # per evitare di passare colonne superflue e non utilizzate alla UDF Pandas.
    # Nota: spostare dati da Spark (JVM) a Pandas (Python) è costoso, quindi è meglio spostare solo i dati necessari.
    input_df = df.select("Hotel_Name", "Review_Date", "Reviewer_Score", "Average_Score").filter(F.col("Total_Number_of_Reviews") >= min_number_of_reviews)

    # Esecuzione GroupBy + ApplyInPandas
    trends = input_df.groupBy("Hotel_Name").applyInPandas(calculate_trend, schema=schema)
    # Nota: ApplyInPandas è un'operazione distribuita, quindi Spark mantiene la parallelizzazione.
    # Per ogni gruppo (Hotel_Name) Spark invia i dati al worker (Python), il quale esegue la UDF Pandas.
    # Spark raccoglie i risultati di tutti questi calcoli paralleli e li unisce in un unico nuovo DataFrame Spark (trends),
    # rispettando la struttura definita in schema.
    return trends

def analyze_tag_influence(df, min_count=50):
    """
    Analizza l'influenza dei tag sul punteggio delle recensioni (MapReduce style).
    
    Map: Esplode la lista dei tag (stringa) in righe singole.
    Reduce: Raggruppa per tag e calcola statistiche (media voto, conteggio).
    
    Args:
        df: DataFrame PySpark con i dati degli hotel
        min_count: Numero minimo di recensioni per tag
    Returns:
        DataFrame PySpark con le colonne selezionate e i top n hotel per nazione
    """
    # 1. Pulizia e Map (Explode)
    # Il campo Tags è una stringa tipo "[' Leisure trip ', ' Couple ', ...]". Dobbiamo rimuovere [ ' ] e splittare per virgola
    clean_tags = F.regexp_replace(F.col("Tags"), "[\\[\\]']", "") # Trasformazione LAZY: restituisce "Leisure trip, Couple, ..."
    splitted_tags = F.split(clean_tags, ",") # Trasformazione LAZY: restituisce ["Leisure trip", "Couple", ...]
    # Explode (Map Phase):
    # aggiunge una nuova colonna Single_Tag e per calcolarne il valore usa explode(splitted_tags)
    # explode(splitted_tags), crea una nuova riga per ogni elemento dell'array splitted_tags
    # Esempio: se una riga ha Tags = "[Leisure Trip, Couple]", explode crea due righe: una con Single_Tag = 'Leisure Trip' e una con Single_Tag = 'Couple'
    # (gli altri campi vengono duplicati dalla riga originale)
    # Quindi si comporta come una FLATMAP: da 1 riga ne crea N (dove N è il numero di tag)
    exploded_df = df.withColumn("Single_Tag", F.explode(splitted_tags))
    # Trim degli spazi bianchi dai tag generati
    exploded_df = exploded_df.withColumn("Single_Tag", F.trim(F.col("Single_Tag")))
    
    # 2. Reduce (GroupBy + Aggregations)
    # Raggruppa per tag e calcola statistiche (media voto, conteggio).
    tag_stats = exploded_df.groupBy("Single_Tag").agg(
        F.count("Reviewer_Score").alias("Count"), # conta le occorrenze di ogni tag
        F.avg("Reviewer_Score").alias("Average_Score"), # calcola la media dei voti per ogni tag
        F.stddev("Reviewer_Score").alias("StdDev_Score") # calcola la deviazione standard dei voti per ogni tag (dev. std. bassa = voti concentrati attorno alla media, alta = voti dispersi)
    )                                                    # se la deviazione standard è alta, significa che i voti sono molto dispersi e non sono affidabili.
    
    # 3. Post-Processing & Filtering
    # Calcoliamo la media globale per confronto
    global_avg = df.select(F.avg("Reviewer_Score")).first()[0]
    # Questo valore serve come punto di riferimento (baseline): se un Tag ha una media di 9.0 e la media globale è 8.5, 
    # allora quel Tag ha un "impatto positivo" (+0.5). Se invece la media del Tag fosse 8.0, avrebbe un "impatto negativo" (-0.5).
    
    # Filtriamo tag poco frequenti e calcoliamo l'impatto (Differenza dalla media globale)
    final_stats = tag_stats.filter(F.col("Count") >= min_count) \
        .withColumn("Impact", F.col("Average_Score") - global_avg) \
        .withColumn("Global_Average", F.lit(global_avg)) # F.lit() permette di creare una colonna con valore costante
    
    # 4. Sorting
    return final_stats.orderBy(F.col("Impact").desc())
