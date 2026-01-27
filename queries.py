from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, DateType
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

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
    # 1. Pulizia e FlatMap (Explode)
    # Il campo Tags è una stringa tipo "[' Leisure trip ', ' Couple ', ...]". Dobbiamo rimuovere [ ' ] e splittare per virgola
    clean_tags = F.regexp_replace(F.col("Tags"), "[\\[\\]']", "") # Trasformazione LAZY: restituisce "Leisure trip, Couple, ..."
    splitted_tags = F.split(clean_tags, ",") # Trasformazione LAZY: restituisce ["Leisure trip", "Couple", ...]
    # Explode (FlatMap Phase):
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
        .withColumn("Global_Average", F.lit(global_avg))

    # --- Calcolo Reliability Index ---
    # Formula euristica: Reliability = (1 / (StdDev + 0.1)) * log(Count)
    # - 1/StdDev premia la stabilità dei voti: dev. std. bassa (voti concentrati attorno alla media) -> reliability alta
    # - +0.1 serve a evitare divisioni per zero e a moderare l'effetto di StdDev bassissime.
    # - log(Count) premia la frequenza dei tag (più un tag è frequente, più è affidabile)
    # - il logaritmo è usato per ridurre l'effetto di Count molto alto (es. log(10) ≈ 1, log(100) ≈ 2, log(1000) ≈ 3)
    final_stats = final_stats.withColumn(
        "Reliability_Index",
        (1 / (F.col("StdDev_Score") + 0.1)) * F.log(F.col("Count"))
    )
    
    # --- Calcolo Impatto Pesato ---
    # Weighted_Impact = Impact * Reliability_Index
    final_stats = final_stats.withColumn(
        "Weighted_Impact",
        F.col("Impact") * F.col("Reliability_Index")
    )
        
    return final_stats.orderBy(F.col("Weighted_Impact").desc())

def analyze_nationality_bias(df, min_reviews=50):
    """
    Analizza il bias dei recensori in base alla nazionalità.
    Raggruppa per nazionalità e calcola statistiche su voti e lunghezza recensioni.
    
    Args:
        df: DataFrame PySpark
        min_reviews: Minimo numero di recensioni per considerare una nazionalità
        
    Returns:
        DataFrame con statistiche per nazionalità
    """
    # 1. Map (riga df -> riga clean_df) e Clean (Rimuoviamo spazi extra e filtriamo eventuali null)
    clean_df = df.withColumn("Reviewer_Nationality", F.trim(F.col("Reviewer_Nationality"))) \
                 .filter(F.col("Reviewer_Nationality").isNotNull()) \
                 .filter(F.col("Reviewer_Nationality") != "")

    # 2. Reduce (GroupBy + Aggregations) (gruppo di righe di clean_df -> riga nation_stats)
    nation_stats = clean_df.groupBy("Reviewer_Nationality").agg(
        F.count("Reviewer_Score").alias("Count"), # conta le occorrenze di ogni nazionalità
        F.avg("Reviewer_Score").alias("Average_Score"), # calcola la media dei voti per ogni nazionalità
        F.avg("Review_Total_Positive_Word_Counts").alias("Avg_Positive_Words"), # calcola la media delle parole positive per ogni nazionalità
        F.avg("Review_Total_Negative_Word_Counts").alias("Avg_Negative_Words") # calcola la media delle parole negative per ogni nazionalità
    )
    
    # 3. Post-Processing & Metrics
    global_avg = df.select(F.avg("Reviewer_Score")).first()[0] # Prendiamo la media globale per confronto
    final_stats = nation_stats.filter(F.col("Count") >= min_reviews) # Filtro per rilevanza statistica (num minimo di recensioni)
    # Calcolo metriche derivate (Deviazione = scostamento dalla media globale, ratio = rapporto tra positive e negative)
    final_stats = final_stats.withColumn("Global_Average", F.lit(global_avg)) \
        .withColumn("Score_Deviation", F.col("Average_Score") - F.col("Global_Average")) \
        .withColumn("Total_Words_Avg", F.col("Avg_Positive_Words") + F.col("Avg_Negative_Words")) \
        .withColumn("Sentiment_Ratio", 
                    F.when(F.col("Total_Words_Avg") > 0, # evita divisione per zero
                           F.col("Avg_Positive_Words") / F.col("Total_Words_Avg"))
                    .otherwise(0.0)
        )
    
    # Ordiniamo per deviazione (dal più generoso al più critico)
    return final_stats.orderBy(F.col("Score_Deviation").desc())

def analyze_local_competitiveness(df, km_radius=2.0, min_competitors=3):
    """
    Analizza la competitività locale (Geospatial Analysis).
    Confronta il punteggio di un hotel con la media dei suoi vicini entro km_radius.
    Identifica "Local Gems" (superano i vicini) e "Underperformers" (sotto la media di zona).
    
    Args:
        df: DataFrame PySpark
        km_radius: Raggio di ricerca in km
        min_competitors: Numero minimo di competitor nel raggio per essere inclusi
    """
    # 1. Preparazione dati (Deduplicazione per hotel)
    hotels = df.select(
        "Hotel_Name", 
        F.col("Average_Score").cast("float"), 
        F.when(F.col("lat") == "NA", None).otherwise(F.col("lat")).cast("double").alias("lat"), 
        F.when(F.col("lng") == "NA", None).otherwise(F.col("lng")).cast("double").alias("lng")
    ).dropDuplicates(["Hotel_Name"]).dropna(subset=["lat", "lng"])

    # 2. Self-Join, calcolo delle distanze, filter per distanza < radius

    # Rinominiamo per distinguere Hotel A (Target) e Hotel B (Neighbor)
    left = hotels.alias("a")
    right = hotels.alias("b")
    joined = left.crossJoin(right).filter(F.col("a.Hotel_Name") != F.col("b.Hotel_Name"))
    
    # Formula Haversine in Spark SQL: serve a calcolare la distanza in linea d'aria tra due punti su una sfera (la Terra)
    #   1. Le coordinate lat e lng sono in gradi, ma la trigonometria funziona in radianti, quindi vanno convertite con F.radians(...)
    #   2. R = 6371 km (Raggio Terra)
    #   3. dLat = rad(lat2 - lat1)                                             Differenza tra latitudini in radianti
    #   4. dLon = rad(lon2 - lon1)                                             Differenza tra longitudini in radianti
    #   5. distance = sin^2(dLat/2) + cos(lat1) * cos(lat2) * sin^2(dLon/2)    Distanza fra due punti su una sfera (adimensionale tra 0 e 1)
    #   6. angle = 2 * asin(sqrt(distance))                                    Distanza angolare in radianti
    #   7. distance_km = R * angle                                             Distanza in km
    joined = joined.withColumn("lat_a_rad", F.radians(F.col("a.lat"))) \
                   .withColumn("lon_a_rad", F.radians(F.col("a.lng"))) \
                   .withColumn("lat_b_rad", F.radians(F.col("b.lat"))) \
                   .withColumn("lon_b_rad", F.radians(F.col("b.lng"))) \
                   .withColumn("dlat", F.col("lat_b_rad") - F.col("lat_a_rad")) \
                   .withColumn("dlon", F.col("lon_b_rad") - F.col("lon_a_rad")) \
                   .withColumn("distance", F.pow(F.sin(F.col("dlat") / 2), 2) + \
                                    F.cos(F.col("lat_a_rad")) * F.cos(F.col("lat_b_rad")) * \
                                    F.pow(F.sin(F.col("dlon") / 2), 2)) \
                   .withColumn("angle", 2 * F.asin(F.sqrt(F.col("distance")))) \
                   .withColumn("distance_km", F.lit(6371.0) * F.col("angle"))
    
    # Filtriamo per distanza < radius
    neighbors = joined.filter(F.col("distance_km") <= km_radius)
    
    # 3. Aggregazione (Reduce)
    # Raggruppiamo per Hotel A e calcoliamo stats dei vicini
    stats = neighbors.groupBy("a.Hotel_Name", "a.Average_Score").agg(
        F.avg("b.Average_Score").alias("Neighborhood_Avg_Score"),
        F.count("b.Hotel_Name").alias("Competitor_Count")
    )
    
    # 4. Post-processing
    final_stats = stats.filter(F.col("Competitor_Count") >= min_competitors) \
                       .withColumn("Score_Delta", F.col("a.Average_Score") - F.col("Neighborhood_Avg_Score")) \
                       .withColumn("Status", 
                                   F.when(F.col("Score_Delta") > 0, "Outperformer")
                                    .otherwise("Underperformer"))
                       
    return final_stats.orderBy(F.col("Score_Delta").desc())

def segment_hotels_kmeans(df, k=4, use_score=True, use_popularity=True, use_verbosity=False, use_location=False, use_nationality=False):
    """
    Esegue la segmentazione degli hotel utilizzando K-Means Clustering.
    Supporta la selezione dinamica delle feature.
    
    Args:
        df: DataFrame PySpark
        k: Numero di cluster
        use_score: Considera Average_Score
        use_popularity: Considera Total_Number_of_Reviews
        use_verbosity: Considera lunghezza recensioni (Avg_Positive_Words, Avg_Negative_Words)
        use_location: Considera lat, lng
        use_nationality: Considera il profilo nazionalità (Top 10 nazioni)
    """
    
    # 1. Base aggregations per Hotel
    # Raggruppiamo per hotel e calcoliamo le feature di base
    # Nota: Usiamo first() su colonne costanti per l'hotel (es. lat, lng, average_score)
    group_cols = [
        F.first("Average_Score").alias("Avg_Score"),
        F.first("Total_Number_of_Reviews").alias("Total_Reviews"),
        F.avg("Review_Total_Positive_Word_Counts").alias("Avg_Pos_Words"),
        F.avg("Review_Total_Negative_Word_Counts").alias("Avg_Neg_Words"),
        F.when(F.first("lat") == "NA", None).otherwise(F.first("lat")).cast("double").alias("Lat"),
        F.when(F.first("lng") == "NA", None).otherwise(F.first("lng")).cast("double").alias("Lng")
    ]
    hotel_features = df.groupBy("Hotel_Name").agg(*group_cols)
    if use_location:
         hotel_features = hotel_features.dropna(subset=["Lat", "Lng"])
    
    # 2. Gestione Feature Opzionali (Nationality Profile)
    if use_nationality:
        # Trova le 10 nazioni più frequenti (in tutto il dataset)
        top_nations = [row['Reviewer_Nationality'] for row in df.groupBy("Reviewer_Nationality").count().orderBy(F.col("count").desc()).limit(10).collect()]
        # Filtra recensioni per le 10 nazioni più frequenti, raggruppa per hotel, 
        # poi fa pivot per nazionalità (crea una colonna per ogni nazionalità)
        # e conta le recensioni per ogni nazionalità
        nat_counts = df.filter(F.col("Reviewer_Nationality").isin(top_nations)) \
                       .groupBy("Hotel_Name") \
                       .pivot("Reviewer_Nationality") \
                       .count() \
                       .na.fill(0) # Sostiuisci null con 0
        
        # Join con le feature base
        hotel_features = hotel_features.join(nat_counts, "Hotel_Name", "left").na.fill(0)

    # 3. Assemblaggio Feature Vector
    input_cols = []
    if use_score: input_cols.append("Avg_Score")
    if use_popularity: input_cols.append("Total_Reviews")
    if use_verbosity: 
        input_cols.append("Avg_Pos_Words")
        input_cols.append("Avg_Neg_Words")
    if use_location:
         # 3D Coordinate Transformation (Unit Sphere)
         # Convertiamo Lat/Lng (gradi) in Radianti per usare funzioni trigonometriche
         # x = cos(lat) * cos(lng)
         # y = cos(lat) * sin(lng)
         # z = sin(lat)
         hotel_features = hotel_features.withColumn("lat_rad", F.radians(F.col("Lat"))) \
                                        .withColumn("lng_rad", F.radians(F.col("Lng"))) \
                                        .withColumn("x", F.cos(F.col("lat_rad")) * F.cos(F.col("lng_rad"))) \
                                        .withColumn("y", F.cos(F.col("lat_rad")) * F.sin(F.col("lng_rad"))) \
                                        .withColumn("z", F.sin(F.col("lat_rad")))
         input_cols.append("x")
         input_cols.append("y")
         input_cols.append("z")
    if use_nationality:
        # Aggiungiamo le colonne delle top nazioni generate dal pivot
        # Nota: dobbiamo recuperare i nomi delle colonne generate (che sono i nomi delle nazioni)
        # Le colonne attuali del df meno quelle base conosciute sono quelle delle nazioni
        base_cols = ["Hotel_Name", "Avg_Score", "Total_Reviews", "Avg_Pos_Words", "Avg_Neg_Words", "Lat", "Lng"]
        nat_cols = [c for c in hotel_features.columns if c not in base_cols]
        input_cols.extend(nat_cols)
    
    if not input_cols:
        raise ValueError("Seleziona almeno una feature per il clustering!")

    # Vector Assembler: unisce le colonne in un unico vettore "features_raw"
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features_raw")
    
    # Standard Scaler: normalizza le feature (media 0, dev.std 1) 
    # Fondamentale per K-Means perchè le distanze euclidee sono sensibili alla scala (es. Reviews=1000 vs Score=10)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    
    # K-Means (# seed=1 per replicabilità)
    kmeans = KMeans(featuresCol="features", k=k, seed=1)
    
    # Pipeline (serve per concatenare le operazioni)
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    
    # Fit & Transform
    model = pipeline.fit(hotel_features) # Fit: apprendimento
    predictions = model.transform(hotel_features) # Transform: predizione
    
    # Ritorniamo il dataframe con le predizioni e tutte le colonne originali
    return predictions

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
    unique_hotels_df = df.select(
        "Hotel_Name", 
        "Hotel_Address", 
        "Average_Score", 
        "Total_Number_of_Reviews",
        F.when(F.col("lat") == "NA", None).otherwise(F.col("lat")).cast("double").alias("lat"),
        F.when(F.col("lng") == "NA", None).otherwise(F.col("lng")).cast("double").alias("lng")
    ).dropDuplicates(["Hotel_Name"])

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
        "Hotel_Address",
        "lat",
        "lng"
    ).orderBy("Nation", F.col("Average_Score").desc())
    # Nota: orderBy() impone questo ordinamento: 
    # 1. per Nation (in ordine alfabetico crescente)
    # 2. per Average_Score (in ordine decrescente)
    return result

def compare_local_vs_tourist_reviews(df, min_reviews_per_group=10):
    """
    Confronta i punteggi dati dai locali (stessa nazione dell'hotel) vs turisti (nazione diversa).
    Identifica "Local Favorites" (prefeiti dai locali) e "Tourist Traps/Favorites" (preferiti dai turisti).
    Inoltre, calcola la composizione delle nazionalità dei visitatori per ciascun hotel (esclude nazionalità con percentuali < 2%).
    
    Args:
        df: DataFrame PySpark
        min_reviews_per_group: Minimo numero di recensioni sia per gruppo Local che Tourist per includere l'hotel
        
    Returns:
        DataFrame con statistiche comparative (Local_Avg, Tourist_Avg, Delta, ecc.)
    """
    
    # 1. Estrazione Nazione Hotel (Stessa logica di get_top_hotels_by_nation)
    # Prende l'ultima parola dell'indirizzo
    df_prep = df.withColumn("Hotel_Nation_Raw", F.element_at(F.split(F.col("Hotel_Address"), " "), -1))
    
    # Mapping correzioni (United Kingdom)
    df_prep = df_prep.withColumn(
        "Hotel_Nation", 
        F.when(F.col("Hotel_Nation_Raw") == "Kingdom", "United Kingdom")
         .otherwise(F.col("Hotel_Nation_Raw"))
    )
    
    # 2. Pulizia Nazionalità Recensore (Trim degli spazi)
    df_prep = df_prep.withColumn("Reviewer_Nationality_Clean", F.trim(F.col("Reviewer_Nationality")))
    # df_prep aggiunge al df originale i campi Hotel_Nation e Reviewer_Nationality_Clean
    
    # 3. Classificazione Review (Local vs Tourist)
    # Local = Recensore della stessa nazione dell'hotel
    df_tagged = df_prep.withColumn(
        "Review_Type",
        F.when(F.col("Hotel_Nation") == F.col("Reviewer_Nationality_Clean"), "Local")
         .otherwise("Tourist")
    )
    
    # 4. Aggregazione (Pivot)
    # Raggruppa per Hotel e Nazione, poi fa Pivot su Review_Type 
    # pivot crea una colonna per ogni Review_Type, cioè una colonna per Local e una per Tourist
    # a quel punto vengono calcolati i valori medi e conteggi per ogni colonna (mantenendo la divisione in gruppi dello stesso hotel)
    # La tabella risultante avrà colonne:
    # Hotel_Name, Hotel_Nation, Local_Avg_Score, Local_Count, Tourist_Avg_Score, Tourist_Count
    stats = df_tagged.groupBy("Hotel_Name", "Hotel_Nation").pivot("Review_Type", ["Local", "Tourist"]).agg(
        F.avg("Reviewer_Score").alias("Avg_Score"),
        F.count("Reviewer_Score").alias("Count")
    )
    
    # 5. Filtri e Metriche Derivate
    final_df = stats.filter(
        (F.col("Local_Count") >= min_reviews_per_group) & 
        (F.col("Tourist_Count") >= min_reviews_per_group)
    )
    
    # Calcolo Delta (Local - Tourist)
    # > 0: Preferito dai Locali
    # < 0: Preferito dai Turisti
    final_df = final_df.withColumn(
        "Preference_Delta", 
        F.col("Local_Avg_Score") - F.col("Tourist_Avg_Score")
    )

    # --- STEP AGGIUNTIVO: Top Nationalities Summary ---
    # Calcolo dei top 10 gruppi nazionali per hotel
    # usiamo df_prep (creato prima), che dispone dei campi Hotel_Nation e Reviewer_Nationality_Clean
    
    # 1. Total Reviews per Hotel (serve per le percentuali)
    hotel_totals = df_prep.groupBy("Hotel_Name").agg(F.count("Reviewer_Nationality_Clean").alias("Total_Reviews"))
    
    # 2. Count per Nation per Hotel
    # restituisce una tabella con Hotel_Name, Reviewer_Nationality_Clean e Nat_Count
    nat_counts = df_prep.groupBy("Hotel_Name", "Reviewer_Nationality_Clean").agg(F.count("*").alias("Nat_Count"))
    
    # 3. Join di nat_counts con hotel_totals per avere il totale e calcolare %
    nat_stats = nat_counts.join(hotel_totals, "Hotel_Name") \
        .withColumn("Pct", (F.col("Nat_Count") / F.col("Total_Reviews") * 100).cast("int"))

    # 4. Filter per escludere nazionalità con percentuali < 2%
    top_nats = nat_stats.filter(F.col("Pct") >= 2)
    
    # 5. Formattazione Stringa "Nation (percentage)"
    top_nats = top_nats.withColumn(
        "Summary", 
        F.concat(F.col("Reviewer_Nationality_Clean"), F.lit(" ("), F.col("Pct"), F.lit("%)"))
    )
    
    # 6. Collect list per avere una sola stringa per hotel
    # Es. "United Kingdom (45%), USA (21%), Australia (10%)"
    nat_summary = top_nats.groupBy("Hotel_Name").agg(
        F.concat_ws(", ", F.collect_list("Summary")).alias("Top_Nationalities")
    )
    
    # 7. Join finale con final_df
    final_df_with_stats = final_df.join(nat_summary, "Hotel_Name", "left")

    # 8. Ordinamento per valore assoluto del delta, per vedere le discrepanze più forti, in senso decrescente
    return final_df_with_stats.orderBy(F.col("Preference_Delta").desc())

def analyze_seasonal_preferences(df, min_reviews=10):
    """
    Analizza le preferenze stagionali e per tipologia di viaggiatore.
    
    Args:
        df: DataFrame PySpark
        min_reviews: Minimo numero di recensioni per segmento (Hotel + Season + Type)
        
    Returns:
        DataFrame con statistiche per Hotel, Stagione e Tipologia Viaggiatore.
    """
    
    # 1. Data Parsing & Enrichment (Map Logic)
    
    # Conversione data per estrarre il mese: Review_Date è stringa 'M/d/yyyy' -> parsing a pyspark.sql.types.DateType
    df_enriched = df.withColumn("Review_Date_Parsed", F.to_date(F.col("Review_Date"), "M/d/yyyy")) \
                    .withColumn("Month", F.month("Review_Date_Parsed"))
    
    # Derivazione Stagione
    # Winter: 12, 1, 2
    # Spring: 3, 4, 5
    # Summer: 6, 7, 8
    # Autumn: 9, 10, 11
    df_enriched = df_enriched.withColumn(
        "Season",
        F.when(F.col("Month").isin(12, 1, 2), "Winter")
         .when(F.col("Month").isin(3, 4, 5), "Spring")
         .when(F.col("Month").isin(6, 7, 8), "Summer")
         .when(F.col("Month").isin(9, 10, 11), "Autumn")
         .otherwise("Unknown")
    )
    
    # Derivazione traveler Types (Multi-label)
    # Un recensore può essere sia "Family" che "Leisure", o "Couple" e "Leisure".
    # Usiamo array() per collezionare tutti i match, e poi explode() per duplicare la riga per ogni match.
    
    # Normalizziamo i tags in minuscolo per case-insensitive matching
    df_enriched = df_enriched.withColumn("Tags_Lower", F.lower(F.col("Tags")))
    
    # Creiamo una colonna array con tutti i tipi trovati
    # Nota: definiamo una lista di regole (Type, Keyword)
    # Se la keyword è presente nei tags, aggiungiamo il Type all'array
    types_expr = F.array(
        F.when(F.col("Tags_Lower").contains("business"), "Business").otherwise(None),
        F.when(F.col("Tags_Lower").contains("family"), "Family").otherwise(None),
        F.when(F.col("Tags_Lower").contains("couple"), "Couple").otherwise(None),
        F.when(F.col("Tags_Lower").contains("solo"), "Solo").otherwise(None),
        F.when(F.col("Tags_Lower").contains("group"), "Group").otherwise(None),
        F.when(F.col("Tags_Lower").contains("leisure"), "Leisure").otherwise(None),
        F.lit("Other")
    )
    
    # explode(types_expr) -> crea una riga per ogni elemento dell'array (duplicando gli altri campi dalla riga originale corrispondente nel DataFrame)
    # filter rimuove le righe con null
    df_exploded = df_enriched.withColumn("Traveler_Type_Raw", F.explode(types_expr)) \
                             .filter(F.col("Traveler_Type_Raw").isNotNull()) \
                             .withColumnRenamed("Traveler_Type_Raw", "Traveler_Type")
    # Se una recensione non ha nessun tag riconosciuto, apparirà solo con il tag "Other"

    # 2. Aggregazione (Reduce Logic): raggruppa per Hotel, Stagione e Tipologia
    stats = df_exploded.groupBy(
        "Hotel_Name", 
        "Hotel_Address", # per conoscere la nazione dell'hotel
        "Season", 
        "Traveler_Type"
    ).agg(
        F.avg("Reviewer_Score").alias("Avg_Score"),
        F.count("Reviewer_Score").alias("Review_Count"),
        F.when(F.first("lat") == "NA", None).otherwise(F.first("lat")).cast("double").alias("lat"), # lat e lng servono per la mappa
        F.when(F.first("lng") == "NA", None).otherwise(F.first("lng")).cast("double").alias("lng")
    )
    
    # 3. Filtering
    final_stats = stats.filter(F.col("Review_Count") >= min_reviews)
    
    # 4. Estrazione Nazione
    final_stats = final_stats.withColumn(
        "Nation_Raw", F.element_at(F.split(F.col("Hotel_Address"), " "), -1)
    ).withColumn(
        "Nation", 
        F.when(F.col("Nation_Raw") == "Kingdom", "United Kingdom")
         .otherwise(F.col("Nation_Raw"))
    ).drop("Nation_Raw")

    # Ordine per Score decrescente
    return final_stats.orderBy(F.col("Avg_Score").desc())

def analyze_stay_duration(df, min_reviews=10):
    """
    Analizza la relazione tra durata del soggiorno e punteggio.
    Estrae il numero di notti dai tag (es. "Stayed 1 night") e raggruppa in categorie.

    Args:
        df: DataFrame PySpark
        min_reviews: Minimo numero di recensioni per categoria/hotel
    Returns:
        DataFrame con statistiche per categoria di durata
    """
    
    # 1. Estrazione Durata Soggiorno (Map Phase: da stringa (tag "Stayed X nights") -> numero di notti)

    clean_tags = F.regexp_replace(F.col("Tags"), "[\\[\\]']", "") # Pulisce la stringa di tags eliminando parentesi quadre [ ] e apici '
    splitted_tags = F.split(clean_tags, ",") # Divide la stringa di tags in singole stringhe ottenendo una colonna di tag singoli
    # explode(splitted_tags) -> crea una riga per ogni tag (duplicando gli altri campi dalla riga originale corrispondente nel DataFrame)
    exploded = df.withColumn("Single_Tag", F.explode(splitted_tags)) \
                 .withColumn("Single_Tag", F.trim(F.col("Single_Tag"))) # trim rimuove spazi vuoti a inizio e fine di ogni tag

    # Filtra le righe che contengono tag come "Stayed 1 night", "Stayed 10 nights", etc. usando la Regex: 'Stayed % night%'
    # Il % finale serve per catturare sia night che nights
    duration_tags = exploded.filter(F.col("Single_Tag").like("Stayed % night%")) 
    
    # Estrazione del numero con F.regexp_extract(column_name, pattern, groupIdx)
    duration_df = duration_tags.withColumn(
        "Nights", 
        F.regexp_extract(F.col("Single_Tag"), r"Stayed (\d+) night", 1).cast("int") # in questo caso accetta sia night che nights
    )
    
    # Filtra eventuali estrazioni fallite (null) o 0 notti
    duration_df = duration_df.filter((F.col("Nights").isNotNull()) & (F.col("Nights") > 0))
    
    # 2. Categorizzazione (Binning) del tipo di soggiorno fra Short: 1-3, Medium: 4-7, Long: 8+
    duration_df = duration_df.withColumn(
        "Stay_Category",
        F.when(F.col("Nights") <= 3, "Short Stay (1-3)")
         .when((F.col("Nights") > 3) & (F.col("Nights") <= 7), "Medium Stay (4-7)")
         .otherwise("Long Stay (8+)")
    )
    
    # 3. Aggregazione per Hotel e Categoria di soggiorno
    stats = duration_df.groupBy("Hotel_Name", "Stay_Category").agg(
        F.avg("Reviewer_Score").alias("Avg_Score"),
        F.count("Reviewer_Score").alias("Review_Count"),
        F.avg("Nights").alias("Avg_Nights_Actual") # Utile per vedere se "Long" è 8 o 20
    )
    
    # 4. Filtering
    final_stats = stats.filter(F.col("Review_Count") >= min_reviews)
    
    return final_stats.orderBy(F.col("Avg_Score").desc())

def analyze_reviewer_experience(df, min_reviews_per_level=5):
    """
    Analizza come cambia il voto in base all'esperienza del recensore.
    Classifica i recensori in:
    - Novice: < 5 recensioni totali
    - Intermediate: 5 - 25 recensioni totali
    - Expert: > 25 recensioni totali
    
    Args:
        df: DataFrame PySpark
        min_reviews_per_level: Minimo numero di recensioni PER LIVELLO per considerare l'hotel
        
    Returns:
        DataFrame con Avg Score per ogni livello e il Gap (Expert - Novice).
    """
    
    # 1. Definizione Categorie Esperienza (Map Phase)
    # Usiamo il campo Total_Number_of_Reviews_Reviewer_Has_Given
    df_exp = df.withColumn(
        "Experience_Level",
        F.when(F.col("Total_Number_of_Reviews_Reviewer_Has_Given") < 5, "Novice")
         .when((F.col("Total_Number_of_Reviews_Reviewer_Has_Given") >= 5) & 
               (F.col("Total_Number_of_Reviews_Reviewer_Has_Given") <= 25), "Intermediate")
         .otherwise("Expert")
    )
    
    # 2. Aggregazione (Reduce Phase)
    # Pivot sui livelli di esperienza: per ogni hotel, per ogni livello di esperienza, calcola il punteggio medio e il numero di recensioni
    stats = df_exp.groupBy("Hotel_Name").pivot("Experience_Level", ["Novice", "Intermediate", "Expert"]).agg(
        F.avg("Reviewer_Score").alias("Avg_Score"), # alias fa diventare {Value}_{Alias} -> Expert_Avg_Score
        F.count("Reviewer_Score").alias("Count") # alias fa diventare {Value}_{Alias} -> Expert_Count
    )
    
    # 3. Filtering e Metriche
    # Vogliamo hotel che abbiano un numero decente di recensioni sia da novizi che da esperti per un confronto sensato
    final_stats = stats.filter(
        (F.col("Novice_Count") >= min_reviews_per_level) & 
        (F.col("Expert_Count") >= min_reviews_per_level)
    ).fillna(0) # Riempie eventuali null
    
    # Calcolo Gap: Se positivo, gli esperti hanno votato più alto dei novizi.
    # Se negativo, gli esperti sono più critici.
    final_stats = final_stats.withColumn(
        "Experience_Gap",
        F.col("Expert_Avg_Score") - F.col("Novice_Avg_Score")
    )
    
    # Aggiungiamo Total Reviews (somma dei count)
    final_stats = final_stats.withColumn(
        "Total_Analyzed_Reviews",
        F.col("Novice_Count") + F.col("Intermediate_Count") + F.col("Expert_Count")
    )
    
    # Ordiniamo per il gap più negativo (gli hotel che deludono di più gli esperti rispetto ai novizi)
    return final_stats.orderBy(F.col("Experience_Gap"))
