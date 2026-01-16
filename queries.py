from pyspark.sql import functions as F
from pyspark.sql.window import Window

def get_top_hotels_by_nation(df, n=10, score_col="Average_Score"):
    """
    Raggruppa gli hotel per nazione e trova gli n migliori per ogni nazione
    basandosi su un criterio di punteggio (default: Average_Score).
    
    Args:
        df: DataFrame PySpark con i dati degli hotel
        n: Numero di hotel da mostrare per ogni nazione
        score_col: La colonna da usare per determinare i "migliori" (modulare)
        
    Returns:
        DataFrame con le colonne selezionate e i top n hotel per nazione
    """
    
    # 1. Estrazione della Nazione
    # L'utente suggerisce che la nazione è l'ultima parola dell'indirizzo.
    # Usiamo split e prendiamo l'ultimo elemento.
    # Gestiamo "United Kingdom" correggendo "Kingdom" -> "United Kingdom".
    
    # Split string by space and get last item
    # Note: element_at with -1 gets the last element in Spark >= 2.4
    df_with_nation = df.withColumn("Nation_Raw", F.element_at(F.split(F.col("Hotel_Address"), " "), -1))
    
    # Clean up common multi-word nations if necessary (mostly United Kingdom in this dataset)
    df_with_nation = df_with_nation.withColumn(
        "Nation", 
        F.when(F.col("Nation_Raw") == "Kingdom", "United Kingdom")
         .otherwise(F.col("Nation_Raw"))
    )
    
    # 2. Logica di Ranking Modulare
    # Definiamo una finestra partizionata per Nazione e ordinata per lo score scelto
    window_spec = Window.partitionBy("Nation").orderBy(F.col(score_col).desc())
    
    # Aggiungiamo il rank
    df_ranked = df_with_nation.withColumn("rank", F.rank().over(window_spec))
    
    # 3. Filtriamo i top n
    top_hotels = df_ranked.filter(F.col("rank") <= n)
    
    # Selezioniamo e ordiniamo per una visualizzazione pulita
    # Includiamo 'Nation', 'Hotel_Name', il punteggio usato, e altre colonne utili
    result = top_hotels.select(
        "Nation",
        "Hotel_Name",
        score_col,
        "Total_Number_of_Reviews",
        "Hotel_Address"
    ).orderBy("Nation", F.col(score_col).desc())
    
    return result
