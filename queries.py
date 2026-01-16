from pyspark.sql import functions as F
from pyspark.sql.window import Window

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
    
    # 3. Logica di Ranking (Top N per Nazione)
    # Definiamo una finestra partizionata per Nazione e ordinata per Average_Score decrescente.
    # Se due hotel hanno lo stesso punteggio, vince chi ha più recensioni (Total_Number_of_Reviews).
    window_spec = Window.partitionBy("Nation").orderBy(F.col("Average_Score").desc(), F.col("Total_Number_of_Reviews").desc())
    # Nota: una Window Function permette di eseguire calcoli su un gruppo di righe correlate alla riga corrente, 
    # senza collassare le righe in una sola (come fa invece groupBy).

    # Aggiungiamo il rank
    df_ranked = df_with_nation.withColumn("rank", F.rank().over(window_spec))
    # Nota: rank() assegna un numero unico a ogni riga all'interno di ogni partizione, 
    # mantenendo l'ordine specificato (in questo caso, decrescente di Average_Score).
    
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
    
    return result
