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
        DataFrame con le colonne selezionate e i top n hotel per nazione
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




def analyze_review_trends(df):
    """
    Analizza il trend temporale dei punteggi delle recensioni per ogni hotel.
    Utilizza una Pandas UDF per calcolare la regressione lineare su ogni gruppo.
    """

    # UDF Pandas che verrà eseguita su ogni gruppo (Hotel)
    # pdf è un pandas DataFrame contenente le recensioni di UN solo hotel
    # Nota: è importante definire una funzione Pandas, che sarà eseguita con applyInPandas,
    # in modo tale che Spark possa mantenere il parallelismo e non dover eseguire la UDF in un unico worker.
    def calculate_trend(pdf): 
        if pdf.empty or len(pdf) < 2: # Non abbastanza dati per un trend, restituisce un DataFrame vuoto
            return pd.DataFrame()
            
        # 1. Ordina per data (fondamentale per il trend temporale) ed elimina le righe con data non valida
        # coerce: converte le date non valide in NaN
        pdf['Review_Date'] = pd.to_datetime(pdf['Review_Date'], format='%m/%d/%Y', errors='coerce')
        pdf = pdf.dropna(subset=['Review_Date'])
        pdf = pdf.sort_values('Review_Date')
        if len(pdf) < 2: # Non abbastanza dati per un trend, restituisce un DataFrame vuoto
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
            "Min_Date": [pdf['Review_Date'].min()],
            "Max_Date": [pdf['Review_Date'].max()]
        })

    # Schema di output della UDF
    schema = StructType([
        StructField("Hotel_Name", StringType(), True),
        StructField("Trend_Slope", FloatType(), True),
        StructField("Review_Count", IntegerType(), True),
        StructField("Average_Score_Calculated", FloatType(), True),
        StructField("Min_Date", DateType(), True),
        StructField("Max_Date", DateType(), True)
    ])

    # Riduce il dataset su cui lavorare, selezionando solo le 3 colonne strettamente necessarie, 
    # per evitare di passare colonne superflue e non utilizzate alla UDF Pandas.
    # Nota: spostare dati da Spark (JVM) a Pandas (Python) è costoso, quindi è meglio spostare solo i dati necessari.
    input_df = df.select("Hotel_Name", "Review_Date", "Reviewer_Score")

    # Esecuzione GroupBy + ApplyInPandas
    trends = input_df.groupBy("Hotel_Name").applyInPandas(calculate_trend, schema=schema)
    # Nota: ApplyInPandas è un'operazione distribuita, quindi Spark mantiene la parallelizzazione.
    # Per ogni gruppo (Hotel_Name) Spark invia i dati al worker (Python), il quale esegue la UDF Pandas.
    # Spark raccoglie i risultati di tutti questi calcoli paralleli e li unisce in un unico nuovo DataFrame Spark (trends),
    # rispettando la struttura definita in schema.
    
    # Aggiunge una colonna "Trend_Description" che spiega l'andamento del trend.
    trends = trends.withColumn(
        "Trend_Description",
        F.when(F.col("Trend_Slope") > 0.0001, "Crescente 📈") # Se slope > 0.0001, trend crescente
         .when(F.col("Trend_Slope") < -0.0001, "Decrescente 📉") # Se slope < -0.0001, trend decrescente
         .otherwise("Stabile ➖") # Se slope è tra -0.0001 e 0.0001, trend stabile
    )
    
    return trends
