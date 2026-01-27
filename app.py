import sys
import os
import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
from pyspark.sql import SparkSession
import time
from queries import get_top_hotels_by_nation, analyze_review_trends, analyze_tag_influence, analyze_nationality_bias, \
    analyze_local_competitiveness, segment_hotels_kmeans, compare_local_vs_tourist_reviews, analyze_seasonal_preferences, \
    analyze_stay_duration, analyze_reviewer_experience

# Fix per "Python worker failed to connect back"
os.environ['PYSPARK_PYTHON'] = sys.executable # impone ai worker di Spark di utilizzare lo stesso interprete Python utilizzato da questo script
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable # impone al driver di Spark di utilizzare lo stesso interprete Python utilizzato da questo script
# Senza questa imposizione, Spark potrebbe tentare di avviare i worker usando un percorso Python di default o errato (es. un'altra versione installata nel sistema)
# Se il "Driver" e i "Worker" non usano esattamente lo stesso interprete Python, non riescono a comunicare tra loro.
# Queste due righe forzano l'uso dello stesso eseguibile per tutti i componenti di Spark (in questo caso lo stesso interprete Python utilizzato dall'ambiente virtuale corrente).

# Configurazione della pagina
st.set_page_config(page_title="Hotel Reviews Analytics", layout="wide")
st.header("📊 Hotel Reviews Analytics con Spark")
st.divider()

# Inizializzazione Spark (Cached per evitare riavvii)
@st.cache_resource
def get_spark_session(app_name="HotelReviewsAnalytics"):
    """Initialize SparkSession"""
    # 'local[*]' usa tutti i core disponibili del computer locale
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN") # per ridurre il numero di messaggi di debug in console
    return spark

# Caricamento dati (Cached)
@st.cache_resource
def get_data(_spark, csv_file="Hotel_Reviews.csv"):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    try:
        df = spark.read.csv(csv_file, header=True, inferSchema=True)
        print(f"Data loaded. Total rows: {df.count()}") # debug
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Main App Logic
try:
    with st.spinner("Inizializzazione Spark..."):
        spark = get_spark_session()
    with st.spinner("Caricamento dati..."):
        df = get_data(spark)
    if df:
        if "get_data_notification_shown" not in st.session_state:
            st.toast(f"Dati caricati con successo! Totale righe: {df.count()}", icon='✅')
            st.session_state.get_data_notification_shown = True
        
        # Sidebar per controlli
        st.sidebar.header("Opzioni Query")
        # Selezione Query
        query_options = {
            "Trend Recensioni (Time Series)": "review_trends",
            "Analisi Influenza Tag": "tag_influence",
            "Analisi Bias Nazionalità": "nationality_bias",
            "Analisi Competitività Locale": "local_competitiveness",
            "Segmentazione Hotel (K-Means)": "hotel_clustering",
            "Migliori Hotel per Nazione": "top_hotels_by_nation",
            "Locals vs Tourists - Preferenze e distribuzione clienti": "local_vs_tourist",
            "Analisi Stagionale & Target": "seasonal_preferences",
            "Analisi Durata Soggiorno": "stay_duration",
            "Analisi Esperienza Recensore": "reviewer_experience",
        }
        selected_query = st.sidebar.radio("Scegli la query da eseguire:", list(query_options.keys()))
        
        st.subheader(f"Query selezionata: {selected_query}")

        # Query: Trend Recensioni
        if query_options[selected_query] == "review_trends":
            st.markdown("""
            Analizza il **trend temporale** dei punteggi di ogni hotel, utilizzando la **Regressione Lineare**.
            
            L'**obiettivo** è capire se un hotel sta migliorando o peggiorando nel tempo.
            
            Viene identificato se la **reputazione** dell'hotel è:
            *   **📈 In crescita**: quando la pendenza della retta di regressione lineare è positiva.
            *   **📉 In decrescita**: quando la pendenza della retta di regressione lineare è negativa.
            *   **➖ Stabile**: quando la pendenza della retta di regressione lineare è zero.

            **Legenda Campi:**
            *   `Trend_Slope`: Pendenza della retta di regressione lineare.
            *   `Review_Count`: Numero totale di recensioni dell'hotel.
            *   `Average_Score_Calculated`: Punteggio medio calcolato sulle recensioni presenti nel dataset.
            *   `Average_Score`: Punteggio medio calcolato su tutte le recensioni ricevute dall'hotel nell'ultimo anno (anche quelle che non sono presenti nel dataset).
            *   `First_Review_Date`: Data della prima recensione dell'hotel.
            *   `Last_Review_Date`: Data della ultima recensione dell'hotel.
            """)
            st.info("**Nota**: Il calcolo del trend viene effettuato tenendo conto solo delle recensioni presenti nel dataset che hanno una data valida. Inoltre, vengono esclusi gli hotel con meno di 30 recensioni valide.")
            min_number_of_reviews = st.slider("Minimo numero di recensioni per albergo", 10, 1000, 30, step=10)
            if st.button("Calcola Trend per tutti gli Hotel"):
                with st.spinner("Calcolo regressione lineare per ogni hotel in corso... (potrebbe richiedere qualche secondo)"):
                    # Esegue la query (da queries.py)
                    trends_df = analyze_review_trends(df, min_number_of_reviews)
                    # Conversione in Pandas per la visualizzazione
                    trends_pdf = trends_df.toPandas()
                    if not trends_pdf.empty:
                        # Top Improving
                        st.subheader("📈 Top 10 Hotel in Crescita")
                        improving = trends_pdf[trends_pdf['Trend_Slope'] > 0].sort_values('Trend_Slope', ascending=False).head(10)
                        st.dataframe(improving[['Hotel_Name', 'Trend_Slope', 'Review_Count', 'Average_Score_Calculated', 'Average_Score', 'First_Review_Date', 'Last_Review_Date']], width='stretch')
                        # Top Declining
                        st.subheader("📉 Top 10 Hotel in Calo")
                        declining = trends_pdf[trends_pdf['Trend_Slope'] < 0].sort_values('Trend_Slope', ascending=True).head(10)
                        st.dataframe(declining[['Hotel_Name', 'Trend_Slope', 'Review_Count', 'Average_Score_Calculated', 'Average_Score', 'First_Review_Date', 'Last_Review_Date']], width='stretch')
                        # Scatter Plot (altair chart) per Trend vs Punteggio Medio
                        st.subheader("Distribuzione Trend vs Punteggio Medio")
                        st.markdown("""
                        In questo grafico è possibile osservare la distribuzione dei trend in relazione al punteggio medio degli hotel.
                        * **^ In alto** si trovano gli hotel con trend positivo (in crescita)
                        * **v In basso** gli hotel con trend negativo (in calo).
                        * **-> A destra** si trovano gli hotel con punteggio medio alto
                        * **<- A sinistra** gli hotel con punteggio medio basso.
                        * **Gli hotel con punteggio medio alto e trend positivo sono i migliori hotel.**  
                        """)
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                        chart = alt.Chart(trends_pdf).mark_circle(size=60).encode(
                            x=alt.X('Average_Score_Calculated', title='Punteggio Medio Calcolato sulle recensioni testuali'),
                            y=alt.Y('Trend_Slope', title='Trend (Slope)'),
                            color=alt.Color('Trend_Slope', scale=alt.Scale(scheme='redblue')),
                            tooltip=['Hotel_Name', 'Trend_Slope', 'Review_Count', 'Average_Score_Calculated', 'Average_Score', 'First_Review_Date', 'Last_Review_Date']
                        ).interactive()
                        st.altair_chart(chart, width='stretch')
                    else:
                        st.warning("Nessun trend calcolato. Verifica i dati.")

        # Query: Tag Influence
        elif query_options[selected_query] == "tag_influence":
            st.markdown("""
            Questa query analizza l'impatto dei tag che appaiono nelle recensioni (es. "double bedroom", "no windows", ecc.), 
            determinando quali **caratteristiche degli hotel** influenzano positivamente le recensioni (sono associate a voti più alti) 
            e quali influenzano negativamente le recensioni (sono associate a voti più bassi).

            Inoltre, per ogni tag viene calcolato un **indice di affidabilità** che tiene conto della frequenza e della deviazione standard dei voti,
            premiando i tag con elevata frequenza e deviazione standard ridotta (cioè i tag che appaiono spesso e in recensioni con voti simili e coerenti fra loro)
            
            1.  **FlatMap**: Esplode la lista dei tag di ogni recensione.
            2.  **Reduce**: Aggrega per tag calcolando voto medio, frequenza e deviazione standard.
            3.  **Analisi di ogni tag**:
                * **Average_Score**: punteggio medio delle recensioni associate al tag.
                * **Count**: frequenza del tag.
                * **Impact**: differenza tra il punteggio medio del tag e il voto medio globale (calcolato su tutte le recensioni).
                * **StdDev_Score**: deviazione standard dei punteggi del tag:
                    * < 1.0: **Molto Attendibile**. C'è forte consenso (quasi tutte le recensioni hanno lo stesso punteggio).
                    * 1.0 - 2.0: **Normale**. C'è una naturale variabilità umana, ma il trend è chiaro
                    * \> 2.0: **Disperso/Controverso**. C'è grande variabilità umana (le recensioni hanno voti molto diversi).
                * **Reliability Index**: Indice euristico `(1/StdDev) * log(Count)` che premia coerenza e frequenza.
                * **Weighted Impact**: `Impact * Reliability Index`.
            """)
            
            min_count = st.slider("Minimo numero di occorrenze per tag", 10, 1000, 50, step=10)
            
            if st.button("Analizza Influenza Tag"):
                with st.spinner(f"Analisi MapReduce sui Tag (filtrando < {min_count} occorrenze)..."):
                    tag_df = analyze_tag_influence(df, min_count=min_count) # da queries.py
                    tag_pdf = tag_df.toPandas() # dataframe da Spark a Pandas per visualizzazione
                    if not tag_pdf.empty:
                        global_avg = tag_pdf['Global_Average'].iloc[0] # legge la media globale dei voti dalla prima riga (è un valore costante)
                        st.metric("Media Globale Voti (calcolata su tutte le recensioni):", f"{global_avg:.2f}")
                        
                        st.subheader("👍 Top 10 Tag Positivi (per Weighted Impact)")
                        st.write("Tag più affidabili che alzano il voto.")
                        pos_tags = tag_pdf[tag_pdf['Impact'] > 0].head(10)
                        st.dataframe(pos_tags[['Single_Tag', 'Average_Score', 'Count', 'Impact', 'Reliability_Index', 'Weighted_Impact']].style.format("{:.2f}", subset=['Average_Score', 'Impact', 'Reliability_Index', 'Weighted_Impact']), width='stretch')
                        
                        st.subheader("👎 Top 10 Tag Negativi (per Weighted Impact)")
                        st.write("Tag più affidabili che abbassano il voto.")
                        neg_tags = tag_pdf[tag_pdf['Impact'] < 0].sort_values('Weighted_Impact', ascending=True).head(10)
                        st.dataframe(neg_tags[['Single_Tag', 'Average_Score', 'Count', 'Impact', 'Reliability_Index', 'Weighted_Impact']].style.format("{:.2f}", subset=['Average_Score', 'Impact', 'Reliability_Index', 'Weighted_Impact']), width='stretch')
                        
                        st.subheader("Grafico: Affidabilità vs Impatto")
                        st.markdown("""
                        * **Asse X (Impact)**: Quanto il tag sposta il voto (Destra=Positivo, Sinistra=Negativo).
                        * **Asse Y (Reliability)**: Quanto è "solido" il dato (Alto=Molto affidabile, Basso=Incerto).
                        * **Obiettivo**: Cerca i tag negli angoli in alto a destra (vincenti sicuri) e in alto a sinistra (problemi certi).
                        """)
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative al tag.")
                        chart = alt.Chart(tag_pdf).mark_circle(size=60).encode(
                            x=alt.X('Impact:Q', title='Impatto (Voto Tag - Media Globale)'),
                            y=alt.Y('Reliability_Index:Q', title='Indice di Affidabilità'),
                            color=alt.Color('Average_Score:Q', scale=alt.Scale(scheme='viridis'), title='Voto Medio'),
                            tooltip=['Single_Tag', 'Average_Score', 'Count', 'Impact', 'Reliability_Index', 'Weighted_Impact', 'StdDev_Score']
                        ).interactive()
                        st.altair_chart(chart, width='stretch')
                        
                    else:
                        st.warning("Nessun tag trovato con i filtri selezionati.")

        # Query: Nationality Bias
        elif query_options[selected_query] == "nationality_bias":
            st.markdown("""
            Questa query cerca di identificare se esistono nazionalità tendenzialmente più generose o severe nei voti.
            Inoltre, calcola un **Sentiment Ratio** basato sul rapporto tra parole positive e totali usate nelle recensioni.

            Legenda campi:
            
            *   **Reviewer Nationality**: Nazionalità del recensore.
            *   **Average Score**: Voto medio assegnato da recensori di stessa nazionalità.
            *   **Count**: Numero di recensioni per nazionalità.
            *   **Score Deviation**: Differenza tra il voto medio della nazionalità e la media globale dei voti di tutte le recensioni.
            *   **Sentiment Ratio**: percentuale di parole positive sul totale delle parole usate nelle recensioni di stessa nazionalità (indicatore di "positività" nel testo).
            *   **Total Words Avg**: Lunghezza media delle recensioni di stessa nazionalità (quanto sono "verbosi").
            """)
            
            min_revs_nat = st.slider("Minimo numero recensioni per nazionalità", 10, 500, 20, step=10)
            
            if st.button("Analizza Bias Nazionalità"):
                with st.spinner("Raggruppamento per nazionalità e calcolo statistiche..."):
                    nat_df = analyze_nationality_bias(df, min_reviews=min_revs_nat) # da queries.py
                    nat_pdf = nat_df.toPandas() # dataframe da Spark a Pandas per visualizzazione
                    
                    if not nat_pdf.empty:
                        global_avg = nat_pdf['Global_Average'].iloc[0] # legge la media globale dei voti dalla prima riga (è un valore costante)
                        st.metric("Media Globale Voti", f"{global_avg:.2f}")
                        
                        st.subheader("😤 I Più Critici (Voti Bassi)")
                        critics = nat_pdf.sort_values("Score_Deviation", ascending=True).head(10) # ordinamento crescente
                        st.dataframe(critics[['Reviewer_Nationality', 'Average_Score', 'Count', 'Score_Deviation', 'Sentiment_Ratio', 'Total_Words_Avg']].style.format("{:.2f}", subset=['Average_Score', 'Score_Deviation', 'Sentiment_Ratio', 'Total_Words_Avg']), width='stretch')
                        
                        st.subheader("🥰 I Più Generosi (Voti Alti)")
                        generous = nat_pdf.sort_values("Score_Deviation", ascending=False).head(10) # ordinamento decrescente
                        st.dataframe(generous[['Reviewer_Nationality', 'Average_Score', 'Count', 'Score_Deviation', 'Sentiment_Ratio', 'Total_Words_Avg']].style.format("{:.2f}", subset=['Average_Score', 'Score_Deviation', 'Sentiment_Ratio', 'Total_Words_Avg']), width='stretch')
                        
                        st.subheader("Correlazione Voto vs Positività Testo (Corenza delle recensioni per nazionalità)")
                        st.write("Chi dà voti alti scrive davvero cose positive? (In alto a destra = Sì, In basso a destra = No)")
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative alla nazionalità.")
                        chart = alt.Chart(nat_pdf).mark_circle().encode(
                            x=alt.X('Average_Score:Q', scale=alt.Scale(domain=[nat_pdf['Average_Score'].min() * 0.95, 10]), title="Voto Medio"),
                            y=alt.Y('Sentiment_Ratio:Q', scale=alt.Scale(domain=[nat_pdf['Sentiment_Ratio'].min() * 0.9, nat_pdf['Sentiment_Ratio'].max() * 1.05]), title="Ratio Parole Positive (Pos/(Total_Words))"),
                            size=alt.Size('Count:Q', legend=None),
                            color=alt.Color('Score_Deviation:Q', scale=alt.Scale(scheme='redblue')),
                            tooltip=['Reviewer_Nationality', 'Average_Score', 'Count', 'Score_Deviation', 'Sentiment_Ratio', 'Total_Words_Avg']
                        ).interactive()
                        st.altair_chart(chart, width='stretch')
                    else:
                        st.warning("Nessuna nazionalità soddisfa i criteri di filtro.")

        # Query: Local Competitiveness
        elif query_options[selected_query] == "local_competitiveness":
            st.markdown("""
            Confronta ogni hotel con i suoi vicini nel raggio di N km, individuando:
            * **Gemme Locali (Outperformers)**: Hotel con punteggio superiore alla media della zona.
            * **Hotel sotto la media (Underperformers)**: Hotel con punteggio inferiore alla media della zona.
            
            Logica di funzionamento:
            1. Calcola la distanza fra ogni hotel e i suoi competitor usando la **Formula di Haversine** (con coordinate geografiche in latitudine e longitudine)
            2. Esclude i competitor che si trovano fuori dal raggio di ricerca specificato
            3. Analizza le seguenti statistiche per ogni hotel:
                * `Average_Score` = Media dei voti dell'hotel (valore basato sulle recensioni raccolte nell'ultimo anno)
                * `Neighborhood_Avg_Score` = Media dei voti dei competitor nella zona
                * `Score_Delta` = Differenza di punteggio tra l'hotel e la **media dei competitor** (se il delta è positivo, l'hotel è meglio dei competitor)
                * `Competitor_Count` = Numero di competitor nella zona
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                km_radius = st.slider("Raggio di ricerca (km)", 0.5, 10.0, 2.0, step=0.5)
            with col2:
                min_competitors = st.number_input("Minimo numero di competitor nella zona", value=5, min_value=1)
                
            if st.button("Analizza Competitività"):
                with st.spinner("Calcolo distanze e analisi di quartiere..."):
                    comp_df = analyze_local_competitiveness(df, km_radius=km_radius, min_competitors=min_competitors) # da queries.py
                    comp_pdf = comp_df.toPandas() # dataframe da Spark a Pandas per visualizzazione
                    
                    if not comp_pdf.empty:
                        st.subheader("💎 Top 10 Gemme Locali (Meglio dei competitor vicini)")
                        gems = comp_pdf[comp_pdf['Score_Delta'] > 0].head(10)
                        st.dataframe(gems[['Hotel_Name', 'Average_Score', 'Neighborhood_Avg_Score', 'Score_Delta', 'Competitor_Count']].style.format("{:.2f}", subset=['Average_Score', 'Neighborhood_Avg_Score', 'Score_Delta']), width='stretch')
                        
                        st.subheader("⚠️ Top 10 Sotto la Media (Peggio dei competitor vicini)")
                        under = comp_pdf[comp_pdf['Score_Delta'] < 0].sort_values("Score_Delta", ascending=True).head(10)
                        st.dataframe(under[['Hotel_Name', 'Average_Score', 'Neighborhood_Avg_Score', 'Score_Delta', 'Competitor_Count']].style.format("{:.2f}", subset=['Average_Score', 'Neighborhood_Avg_Score', 'Score_Delta']), width='stretch')
                        
                        st.markdown("""
                        ## Grafico: Performance Relativa

                        Confronto tra il voto dell'hotel (Asse X) e il voto medio della zona (Asse Y).
                        * **Sotto la diagonale**: L'hotel è meglio della zona (Gemma Locale)
                        * **Sopra la diagonale**: L'hotel è peggio della zona (Underperformer)
                        """)
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                        chart = alt.Chart(comp_pdf).mark_circle(size=60).encode(
                            x=alt.X('Average_Score:Q', title='Voto Hotel', scale=alt.Scale(domain=[6, 10])),
                            y=alt.Y('Neighborhood_Avg_Score:Q', title='Media Voto Zona (Vicini)', scale=alt.Scale(domain=[6, 10])),
                            color=alt.condition(
                                alt.datum.Score_Delta > 0,
                                alt.value("green"),
                                alt.value("red")
                            ),
                            tooltip=['Hotel_Name', 'Average_Score', 'Neighborhood_Avg_Score', 'Competitor_Count', 'Score_Delta']
                        ).interactive()
                        
                        # Aggiungiamo la linea diagonale per riferimento
                        line = alt.Chart(pd.DataFrame({'x': [6, 10], 'y': [6, 10]})).mark_line(color='gray', strokeDash=[5, 5]).encode(x='x', y='y')
                        
                        st.altair_chart(chart + line, width='stretch')
                        
                    else:
                        st.warning(f"Nessun hotel trovato con almeno {min_competitors} competitor nel raggio di {km_radius} km.")
        
        # Query: Hotel Clustering
        elif query_options[selected_query] == "hotel_clustering":
            st.markdown("""
            🎯 Identifica gruppi di hotel simili in base alle loro caratteristiche. Questo metodo *unsupervised* permette di scoprire pattern nascosti nel dataset.
            
            **Scegli le caratteristiche da utilizzare per il raggruppamento:**
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                k_clusters = st.slider("Numero di Cluster (gruppi)", 2, 10, 4)
                
            st.write("**Feature Selezionate:**")
            c1, c2, c3, c4, c5 = st.columns(5)
            use_score = c1.checkbox("Punteggio (Avg Score)", value=True)
            use_popularity = c2.checkbox("Popolarità (Num. Recensioni)", value=True)
            use_verbosity = c3.checkbox("Verbosità (Lunghezza Recensioni)", value=True)
            use_location = c4.checkbox("Posizione (Lat/Lng)", value=True)
            use_nationality = c5.checkbox("Profilo Nazionalità", value=True, help="Includi la % di provenienza dei clienti (Top 10 nazioni)")

            if "clustering_pdf" not in st.session_state: # per non ricalcolare il dataframe ogni volta che si aggiorna la pagina
                st.session_state["clustering_pdf"] = None

            if st.button("Esegui Segmentazione"):
                if not any([use_score, use_popularity, use_verbosity, use_location, use_nationality]):
                    st.error("Seleziona almeno una feature!")
                else:
                    with st.spinner(f"Addestramento modello K-Means (k={k_clusters}) su Spark..."):
                        clustered_df = segment_hotels_kmeans(
                            df, 
                            k=k_clusters,
                            use_score=use_score,
                            use_popularity=use_popularity,
                            use_verbosity=use_verbosity,
                            use_location=use_location,
                            use_nationality=use_nationality
                        )
                        # Rimuoviamo le colonne vettoriali di Spark ML (features, features_raw) prima di convertire in Pandas
                        # Altrimenti PyArrow/Streamlit crashano perché non sanno serializzare i DenseVector
                        pdf = clustered_df.drop("features", "features_raw").toPandas()
                        st.session_state["clustering_pdf"] = pdf
                        # Mostriamo messaggio di successo temporaneo
                        msg_placeholder = st.empty()
                        msg_placeholder.success(f"Analisi completata! Hotel suddivisi in {k_clusters} cluster.")
                        time.sleep(3)
                        msg_placeholder.empty()
            
            # Se abbiamo risultati in memoria (o appena calcolati), mostriamo la visualizzazione
            if st.session_state["clustering_pdf"] is not None:
                pdf = st.session_state["clustering_pdf"]
                
                # --- 1. Statistiche Cluster ---
                st.subheader("📊 Analisi dei Gruppi Identificati")
                
                # Seleziona solo colonne numeriche rilevanti per la visualizzazione (e controlla se le colonne opzionali sono presenti)
                numeric_cols = ["Avg_Score", "Total_Reviews"]
                if "Avg_Pos_Words" in pdf.columns: numeric_cols.extend(["Avg_Pos_Words", "Avg_Neg_Words"])
                if "Lat" in pdf.columns: numeric_cols.extend(["Lat", "Lng"])
                
                # Calcola la media delle colonne numeriche per ogni cluster per interpretare il risultato
                cluster_stats = pdf.groupby("prediction")[numeric_cols].mean().reset_index()
                cluster_counts = pdf['prediction'].value_counts().reset_index() # conta quanti hotel ci sono in ogni cluster
                cluster_counts.columns = ['prediction', 'Count'] # rinomina la colonna prediction in Count
                
                summary = pd.merge(cluster_stats, cluster_counts, on="prediction") # merge delle due tabelle
                summary['Cluster Name'] = summary['prediction'].apply(lambda x: f"Cluster {x}") # rinomina la colonna prediction in Cluster Name
                
                # Filtriamo le colonne da visualizzare
                desired_cols = ["Cluster Name", "Count", "Avg_Score", "Total_Reviews", "Avg_Pos_Words", "Avg_Neg_Words"]
                final_cols = [c for c in desired_cols if c in summary.columns]

                # Palette fissa (RGB) per max 10 cluster
                CLUSTER_COLORS = [
                    [255, 0, 0, 160],   # Rosso
                    [0, 255, 0, 160],   # Verde
                    [0, 0, 255, 160],   # Blu
                    [255, 165, 0, 160], # Arancione
                    [128, 0, 128, 160], # Viola
                    [0, 255, 255, 160], # Ciano
                    [255, 0, 255, 160], # Magenta
                    [255, 255, 0, 160], # Giallo
                    [128, 128, 128, 160], # Grigio
                    [0, 191, 255, 160]      # Celeste
                ]

                # Funzione per colorare le celle
                def color_cluster_name(val):
                    try:
                        cluster_id = int(str(val).split(" ")[1])
                        c = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
                        # CSS rgba usa 0-1 per alpha
                        return f'background-color: rgba({c[0]}, {c[1]}, {c[2]}, {c[3]/255:.2f}); color: white'
                    except:
                        return ''

                # Formattazione colonne
                format_dict = {
                    "Total_Reviews": "{:.0f}",
                    "Count": "{:.0f}",
                    "Avg_Score": "{:.2f}",
                    "Avg_Pos_Words": "{:.1f}",
                    "Avg_Neg_Words": "{:.1f}"
                }

                # Mostriamo la tabella
                st.dataframe(summary[final_cols].style
                             .background_gradient(cmap='Blues', subset=['Count'])
                             .map(color_cluster_name, subset=['Cluster Name'])
                             .format(format_dict), width='stretch')
                
                # --- 2. Mappa Geografica (Se attiva Posizione) ---
                if use_location:
                    st.subheader("🗺️ Distribuzione Geografica Cluster")
                    st.write("Visualizza come i cluster sono distribuiti geograficamente.")
                    st.info("Nota: cliccando su un punto della mappa è possibile visualizzare le informazioni relative al cluster.")
                    # Applichiamo il colore al dataframe Pandas
                    # Creiamo una colonna 'color' contenente la lista [R, G, B, A]
                    pdf["color"] = pdf["prediction"].apply(lambda pid: CLUSTER_COLORS[pid % len(CLUSTER_COLORS)])
                    
                    # 2. Configurazione View State (Dove guardare all'inizio)
                    view_state = pdk.ViewState(
                        latitude=pdf["Lat"].mean(),
                        longitude=pdf["Lng"].mean(),
                        zoom=4,
                        pitch=0,
                    )
                    
                    # 3. Configurazione Layer (Scatterplot)
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=pdf,
                        get_position='[Lng, Lat]',
                        get_color='color',
                        get_radius=10000, # Raggio in metri (10km per essere visibili a livello continentale)
                        pickable=True,    # Permette di cliccare/hover
                        opacity=0.8,
                        filled=True,
                        radius_min_pixels=5,
                        radius_max_pixels=50,
                    )
                    
                    # 4. Render Mappa
                    st.pydeck_chart(pdk.Deck(
                        map_style=None, # Usa stile default (CartoDB Dark)
                        initial_view_state=view_state,
                        layers=[layer],
                        tooltip={
                            "html": "<b>Hotel:</b> {Hotel_Name}<br/><b>Cluster:</b> {prediction}<br/><b>Score:</b> {Avg_Score}",
                            "style": {"color": "white"}
                        }
                    ))

                # --- 3. Scatter Plot Interattivo (Performance) ---
                st.subheader("📍 Esplorazione Performance (Scatter Plot)")
                st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                # Opzioni assi dinamiche (Escludiamo Lat/Lng)
                avail_cols = ["Avg_Score", "Total_Reviews"]
                if "Avg_Pos_Words" in pdf.columns: avail_cols.extend(["Avg_Pos_Words", "Avg_Neg_Words"])
                
                ac1, ac2 = st.columns(2)
                x_axis = ac1.selectbox("Asse X", avail_cols, index=0)
                y_axis = ac2.selectbox("Asse Y", avail_cols, index=1 if len(avail_cols) > 1 else 0)
                
                chart = alt.Chart(pdf).mark_circle(size=60).encode(
                    x=alt.X(x_axis, scale=alt.Scale(zero=False)),
                    y=alt.Y(y_axis, scale=alt.Scale(zero=False)),
                    color=alt.Color('prediction:N', title='Cluster', scale=alt.Scale(scheme='category10')),
                    tooltip=['Hotel_Name', 'prediction', 'Avg_Score', 'Total_Reviews']
                ).interactive()
                
                st.altair_chart(chart, width='stretch')
                
                # --- 4. Analisi Nazionalità (Se attivata) ---
                if use_nationality:
                    st.subheader("🌍 Composizione Etnica per Cluster")
                    st.write("Come si distribuiscono le nazionalità nei vari gruppi?")
                    
                    # Recuperiamo le colonne delle nazionalità (tutte quelle non standard)
                    base_cols = ["Hotel_Name", "prediction", "features", "features_raw", 
                                 "Avg_Score", "Total_Reviews", "Avg_Pos_Words", "Avg_Neg_Words", "Lat", "Lng",
                                 "color", "lat_rad", "lng_rad", "x", "y", "z"]
                    nat_cols = [c for c in pdf.columns if c not in base_cols]
                    
                    if nat_cols:
                        # Conversione esplicita a numerico per evitare errori di tipo 'object' (dovuti forse agli spazi nei nomi colonne)
                        for c in nat_cols:
                            pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0)
                            
                        # Calcoliamo la media di ogni nazionalità per cluster
                        nat_summary = pdf.groupby("prediction")[nat_cols].mean().reset_index()
                        # Melt per visualizzazione (Cluster, Nation, Percentage)
                        nat_melted = nat_summary.melt(id_vars="prediction", var_name="Nation", value_vars=nat_cols, value_name="Avg_Reviews_Count")
                        
                        # Grafico Stacked Bar
                        nat_chart = alt.Chart(nat_melted).mark_bar().encode(
                            x=alt.X('prediction:N', title='Cluster'),
                            y=alt.Y('Avg_Reviews_Count:Q', stack='normalize', title='Distribuzione Nazionalità'),
                            color=alt.Color('Nation:N', scale=alt.Scale(scheme='tableau20')),
                            tooltip=['prediction', 'Nation', 'Avg_Reviews_Count']
                        ).properties(height=400)
                        
                        st.altair_chart(nat_chart, width='stretch')

        # Query: Top Hotels by Nation
        elif query_options[selected_query] == "top_hotels_by_nation":
            st.write("Questa query consente di individuare i **migliori hotel per ogni nazione**. Il criterio di ranking utilizzato è il **punteggio medio** ottenuto nelle recensioni dei clienti, inoltre in caso di parità si predilige l'hotel con il **numero totale di recensioni** più elevato.")
            n_per_nation = st.number_input("Numero di migliori hotel da visualizzare per ogni nazione:", min_value=1, max_value=50, value=3)
            if st.button("Analizza per Nazione"):
                 with st.spinner("Analisi dataset in corso..."):
                    # Esegui la query (da queries.py)
                    nation_results = get_top_hotels_by_nation(df, n=n_per_nation)
                    # Converti a Pandas (pdf = pandas dataframe) e visualizza risultati
                    nation_pdf = nation_results.toPandas()
                    st.write(f"### Top {n_per_nation} Hotel per Nazione")
                    st.dataframe(nation_pdf, width='stretch')
                    # Grafico a barre per confrontare i punteggi
                    if not nation_pdf.empty:
                        st.write("#### Distribuzione dei Punteggi degli Hotel")
                        chart = alt.Chart(nation_pdf).mark_bar().encode(
                            x=alt.X('Average_Score:Q', title='Punteggio Medio', scale=alt.Scale(domain=[nation_pdf['Average_Score'].min()*0.9, 10])),
                            y=alt.Y('Hotel_Name:N', sort='-x', title='Hotel'), # -x ordina gli hotel dall'alto al basso in base al valore dell'asse X, in ordine decrescente
                            color='Nation:N',
                            tooltip=['Nation', 'Hotel_Name', 'Average_Score']
                        ).interactive()
                        st.altair_chart(chart, width='stretch')
                        
                        # --- Mappa Geografica ---
                        st.subheader("🗺️ Mappa dei Migliori Hotel")
                        st.info("Nota: cliccando su un punto della mappa è possibile visualizzare le informazioni relative all'hotel.")
                        # Filtra hotel con coordinate valide (dropna su lat/lng)
                        map_df = nation_pdf.dropna(subset=['lat', 'lng'])
                        
                        if not map_df.empty:
                            st.write(f"Visualizzazione geografica di {len(map_df)} hotel (quelli con coordinate valide).")
                            
                            # View State centrato sulla media delle coordinate
                            view_state = pdk.ViewState(
                                latitude=map_df["lat"].mean(),
                                longitude=map_df["lng"].mean(),
                                zoom=3,
                                pitch=0,
                            )
                            
                            # Layer Scatterplot
                            layer = pdk.Layer(
                                "ScatterplotLayer",
                                data=map_df,
                                get_position='[lng, lat]',
                                get_color='[255, 215, 0, 200]', # Oro/Giallo
                                get_radius=30000, # 30km di raggio per essere visibili
                                pickable=True,
                                opacity=0.8,
                                filled=True,
                                radius_min_pixels=5,
                                radius_max_pixels=50,
                            )
                            
                            st.pydeck_chart(pdk.Deck(
                                map_style=None,
                                initial_view_state=view_state,
                                layers=[layer],
                                tooltip={
                                    "html": "<b>{Hotel_Name}</b><br/>{Nation}<br/>Score: {Average_Score}",
                                    "style": {"color": "white"}
                                }
                            ))
                        else:
                            st.warning("Nessuna coordinata valida trovata per gli hotel selezionati.")
        
        # Query: Local vs Tourist
        elif query_options[selected_query] == "local_vs_tourist":
            st.markdown("""
            #### 🏠 Locals' Favorites vs 📸 Tourist Traps
            
            Analizza come cambia la percezione degli hotel tra chi è del posto (Locals) e i turisti internazionali (Tourists).
            
            *   **Locals (Recensori Locali)**: Recensori che hanno la stessa nazionalità della nazione in cui si trova l'hotel.
            *   **Tourists (Recensori Turisti)**: Recensori di nazionalità diversa.
            
            **Indicatori:**
            *   **Local_Avg_Score**: Voto medio dato dai locali.
            *   **Tourist_Avg_Score**: Voto medio dato dai turisti.
            *   **Preference_Delta** (`Local - Tourist`): 
                *   Valori **POSITIVI** (> 0): L'hotel piace più ai Locali (possibile "Gemma Nascosta" o esperienza autentica).
                *   Valori **NEGATIVI** (< 0): L'hotel piace più ai Turisti (possibile "Tourist Trap" o standard internazionale generico).

            #### 🌍 Distribuzione Nazionalità
            Visualizza graficamente la composizione delle nazionalità dei visitatori per ciascun hotel.
            """)
            
            min_revs_group = st.slider("Minimo recensioni per gruppo (Local & Tourist)", 5, 100, 20, help="Esclude hotel con pochi dati per uno dei due gruppi")
            
            if st.button("Analizza Preferenze"):
                with st.spinner("Confronto opinioni Locali vs Turisti..."):
                    comp_df = compare_local_vs_tourist_reviews(df, min_reviews_per_group=min_revs_group) # da queries.py
                    comp_pdf = comp_df.toPandas() # da Spark a Pandas per la visualizzazione
                    
                    if not comp_pdf.empty:
                        # KPI Globali (KPI = Key Performance Indicator)
                        avg_delta = comp_pdf['Preference_Delta'].mean()
                        st.metric("Discrepanza Media (Local - Tourist)", f"{avg_delta:+.2f}", help="Se positivo, in media i locali sono più generosi dei turisti.")
                        
                        st.subheader("🏠 Hotel preferiti dai Locali (Rispetto ai Turisti)")
                        st.write("Hotel dove il voto dei locali supera di più quello dei turisti.")
                        local_favs = comp_pdf.sort_values("Preference_Delta", ascending=False).head(10)
                        st.dataframe(local_favs[['Hotel_Name', 'Hotel_Nation', 'Local_Avg_Score', 'Tourist_Avg_Score', 'Preference_Delta', 'Top_Nationalities']].style.format("{:.2f}", subset=['Local_Avg_Score', 'Tourist_Avg_Score', 'Preference_Delta']).background_gradient(subset=['Preference_Delta'], cmap='Greens'), width='stretch')
                            
                        st.subheader("📸 Hotel preferiti dai Turisti (Rispetto ai Locali)")
                        st.write("Hotel dove il voto dei turisti supera di più quello dei locali.")
                        tourist_favs = comp_pdf.sort_values("Preference_Delta", ascending=True).head(10)
                        st.dataframe(tourist_favs[['Hotel_Name', 'Hotel_Nation', 'Local_Avg_Score', 'Tourist_Avg_Score', 'Preference_Delta', 'Top_Nationalities']].style.format("{:.2f}", subset=['Local_Avg_Score', 'Tourist_Avg_Score', 'Preference_Delta']).background_gradient(subset=['Preference_Delta'], cmap='Reds_r'), width='stretch')
                        
                        st.subheader("Grafico: Discrepanza Voti")
                        st.write("Confronto diretto tra Voto Local (Y) e Voto Tourist (X).")
                        st.write("*   **Punti Sopra la diagonale**: Meglio per i Locali.")
                        st.write("*   **Punti Sotto la diagonale**: Meglio per i Turisti.")
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                        # Scatter Plot
                        chart = alt.Chart(comp_pdf).mark_circle(size=60).encode(
                            x=alt.X('Tourist_Avg_Score:Q', title='Voto Turisti', scale=alt.Scale(domain=[6, 10])),
                            y=alt.Y('Local_Avg_Score:Q', title='Voto Locali', scale=alt.Scale(domain=[6, 10])),
                            color=alt.Color('Preference_Delta:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0), title='Delta (Local-Tourist)'),
                            tooltip=['Hotel_Name', 'Hotel_Nation', 'Local_Avg_Score', 'Tourist_Avg_Score', 'Preference_Delta', 'Local_Count', 'Tourist_Count']
                        ).interactive()
                        # Diagonale
                        line = alt.Chart(pd.DataFrame({'x': [6, 10], 'y': [6, 10]})).mark_line(color='gray', strokeDash=[5, 5]).encode(x='x', y='y')
                        st.altair_chart(chart + line, width='stretch')

                        # --- Stacked Bar Chart Nazionalità ---
                        st.markdown("""
                        #### 🌍 Distribuzione Nazionalità

                        Visualizza la composizione percentuale delle nazionalità dei visitatori degli Hotel per ciascuna categoria.

                        Sono escluse le nazionalità che rappresentano meno del 2% dei visitatoridi un hotel.
                        La non totalità delle percentuali per un hotel può dipendere sia da questa esclusione arbitraria,
                        sia dall'eventuale mancanza di dati sulla nazionalità dei visitatori.

                        """)
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative ad hotel e distribuzione nazionalità.")
                        # Helper per parsare la stringa "Nation (XX%), Nation (YY%)"
                        def parse_nationalities(row, category):
                            if not row['Top_Nationalities']: return []
                            items = str(row['Top_Nationalities']).split(", ")
                            parsed = []
                            for item in items:
                                try:
                                    # Expect format: "Nation Name (XX%)"
                                    name, pct = item.rsplit(" (", 1)
                                    pct = float(pct.rstrip("%)"))
                                    parsed.append({
                                        "Hotel_Name": row['Hotel_Name'],
                                        "Nation": name,
                                        "Percentage": pct,
                                        "Category": category
                                    })
                                except:
                                    continue
                            return parsed

                        # Preparazione Dati per Grafico
                        chart_data = []
                        for _, row in local_favs.iterrows():
                            chart_data.extend(parse_nationalities(row, "Local Favorites"))
                        for _, row in tourist_favs.iterrows():
                            chart_data.extend(parse_nationalities(row, "Tourist Favorites"))
                            
                        if chart_data:
                            nat_chart_df = pd.DataFrame(chart_data)
                            
                            # Grafico Stacked Bar diviso per Categoria (Local Favs vs Tourist Favs)
                            bars = alt.Chart(nat_chart_df).mark_bar().encode(
                                y=alt.Y('Hotel_Name:N', sort='-x', title='Hotel'),
                                x=alt.X('Percentage:Q', title='Percentuale Visitatori'),
                                color=alt.Color('Nation:N', scale=alt.Scale(scheme='tableau20'), title='Nazionalità'),
                                tooltip=['Hotel_Name', 'Nation', 'Percentage']
                            )
                            
                            st.altair_chart(bars.facet(
                                row=alt.Row('Category:N', title=None, header=alt.Header(titleOrient="top", labelOrient="top", labelFontSize=14, labelFontWeight='bold'))
                            ).resolve_scale(y='independent'), width='stretch')
                        else:
                            st.warning("Dati sulle nazionalità non disponibili per il grafico.")

                    else:
                        st.warning(f"Nessun hotel trovato con almeno {min_revs_group} recensioni per entrambi i gruppi.")

        # Query: Seasonal Preferences
        elif query_options[selected_query] == "seasonal_preferences":
            st.markdown("""
            #### 🍂👨‍👩‍👧‍👦 Analisi Stagionale e Profilazione Viaggiatore
            
            Scopri **quando** andare e **con chi**. Questa query analizza come cambiano le performance degli hotel in base alla **Stagione** e al **Tipo di Viaggiatore**.
            
            *   **Season**: Inverno, Primavera, Estate, Autunno (derivata dalla data delle recensioni).
            *   **Traveler Type**: Families, Couples, Business, Solo, Group (derivato dai Tag).
            """)
            
            # Filtri
            col_seasons, col_types = st.columns(2)
            
            with col_seasons:
                st.markdown("**Seleziona Stagioni:**")
                seasons_list = ["Winter", "Spring", "Summer", "Autumn"]
                selected_seasons = []
                for s_name in seasons_list:
                    if st.checkbox(s_name, value=True, key=f"chk_s_{s_name}"):
                        selected_seasons.append(s_name)
            
            with col_types:
                st.markdown("**Seleziona Tipo Viaggiatore:**")
                types_list = ["Family", "Couple", "Business", "Solo", "Group", "Leisure"]
                selected_types = []
                for t_name in types_list:
                    if st.checkbox(t_name, value=True, key=f"chk_t_{t_name}"):
                        selected_types.append(t_name)
                
            min_revs_seas = st.slider("Minimo recensioni per includere un hotel nella ricerca:", 5, 50, 10)
            
            if "seasonal_pdf" not in st.session_state:
                st.session_state["seasonal_pdf"] = None

            if st.button("Analizza Stagionalità"):
                with st.spinner("Calcolo metriche stagionali per segmento..."):
                    # Esecuzione Query PySpark
                    seasonal_df = analyze_seasonal_preferences(df, min_reviews=min_revs_seas)
                    pdf = seasonal_df.toPandas()
                    st.session_state["seasonal_pdf"] = pdf
            
            # Recuperiamo i dati dalla sessione (se presenti)
            if st.session_state["seasonal_pdf"] is not None:
                pdf = st.session_state["seasonal_pdf"]
                
                if not pdf.empty:
                    # Filtering in Pandas (per visualizzazione) 
                    display_df = pdf.copy()

                    # Nessuna selezione = "Mostra Tutto" (inclusi i valori nascosti come "Other"). 
                    # Selezione attiva = "Mostra solo le righe che soddisfano i campi spuntati".

                    # Filtra Stagioni
                    if selected_seasons: # se l'utente non ha selezionato nessun campo, questo filtraggio viene saltato
                        display_df = display_df[display_df['Season'].isin(selected_seasons)]
                    
                    # Filtra Tipi Viaggiatore
                    if selected_types: # se l'utente non ha selezionato nessun campo, questo filtraggio viene saltato
                        display_df = display_df[display_df['Traveler_Type'].isin(selected_types)]
                        
                    # Formatta le stringhe del titolo
                    # Prende la lista di opzioni selezionate (che è una lista di stringhe, ad esempio ['Winter', 'Summer']) 
                    # e le unisce in un'unica stringa separata da virgole.
                    # Se non ci sono opzioni selezionate, restituisce "All".
                    season_str = ", ".join(selected_seasons) if selected_seasons else "All Seasons"
                    type_str = ", ".join(selected_types) if selected_types else "All Traveler Types"
                    
                    # 1. Top Hotel Table
                    st.subheader(f"🏆 Top Hotel: {season_str} & {type_str}")
                    st.write("I migliori hotel per la combinazione selezionata.")
                    
                    top_hotels = display_df.sort_values("Avg_Score", ascending=False).head(20)
                    st.dataframe(top_hotels[['Hotel_Name', 'Season', 'Traveler_Type', 'Avg_Score', 'Review_Count', 'Nation']].style.format("{:.2f}", subset=['Avg_Score']), width='stretch')
                    
                    # 2. Heatmap Hotel Specifico
                    st.divider()
                    st.subheader("🌡️ Heatmap Stagionale Hotel")
                    st.write("Seleziona un hotel per vedere come performa in tutte le stagioni e con tutti i tipi di viaggiatori.")
                    
                    # Lista hotel unici dai risultati (ordinati per score generale nel filtro corrente, o globale se All)
                    unique_hotels = display_df['Hotel_Name'].unique()
                    if len(unique_hotels) > 0:
                        hotel_to_inspect = st.selectbox("Scegli Hotel da analizzare", unique_hotels)
                        
                        # Filtriamo il PDF originale (completo) per quell'hotel
                        hotel_stats = pdf[pdf['Hotel_Name'] == hotel_to_inspect]
                        
                        # Heatmap Chart
                        # X = Season (Ordiniamo logicamente)
                        # Y = Traveler Type
                        # Color = Avg_Score
                        
                        base = alt.Chart(hotel_stats).encode(
                            x=alt.X('Season:N', sort=["Winter", "Spring", "Summer", "Autumn"], title="Stagione"),
                            y=alt.Y('Traveler_Type:N', title="Tipo Viaggiatore")
                        )
                        
                        heatmap = base.mark_rect().encode(
                            color=alt.Color('Avg_Score:Q', scale=alt.Scale(domain=[6, 10], scheme='redyellowgreen'), title="Voto Medio"),
                            tooltip=['Season', 'Traveler_Type', 'Avg_Score', 'Review_Count']
                        )
                        
                        text = base.mark_text(baseline='middle').encode(
                            text=alt.Text('Avg_Score', format='.1f'),
                            color=alt.value('black') # O dinamico in base al background
                        )
                        
                        st.altair_chart(heatmap + text, width='stretch')
                        
                    # 3. Mappa Top Results
                    st.subheader(f"🗺️ Mappa dei migliori risultati ({season_str} - {type_str})")
                    st.info("Nota: cliccando su un punto della mappa è possibile visualizzare le informazioni relative all'hotel.")
                    # Usiamo top_hotels filtrato
                    map_df = top_hotels.dropna(subset=['lat', 'lng'])
                    if not map_df.empty:
                        view_state = pdk.ViewState(
                            latitude=map_df["lat"].mean(),
                            longitude=map_df["lng"].mean(),
                            zoom=4
                        )
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=map_df,
                            get_position='[lng, lat]',
                            get_color='[0, 128, 255, 200]',
                            get_radius=10000,
                            pickable=True,
                            opacity=0.8,
                            filled=True,
                            radius_min_pixels=5,
                            radius_max_pixels=50,
                        )
                        st.pydeck_chart(pdk.Deck(
                            map_style=None,
                            initial_view_state=view_state,
                            layers=[layer],
                            tooltip={
                                "html": "<b>{Hotel_Name}</b><br/>Score: {Avg_Score}<br/>Type: {Traveler_Type}<br/>Season: {Season}",
                                "style": {"color": "white"}
                            }
                        ))
                        
                else:
                    st.warning("Nessun dato trovato con i criteri minimi selezionati.")
        
        # Query: Stay Duration
        elif query_options[selected_query] == "stay_duration":
            st.markdown("""
            #### ⏳ Analisi dell'impatto della Durata del Soggiorno
            
            Analizza come varia la soddisfazione dei clienti in base alla durata del loro soggiorno.
            Le durate sono categorizzate in:
            *   **Short Stay**: 1-3 notti
            *   **Medium Stay**: 4-7 notti
            *   **Long Stay**: 8+ notti
            
            Questa analisi aiuta a capire quali hotel sono più adatti a toccate e fuga o a lunghe vacanze.
            """)
            
            min_revs_stay = st.slider("Minimo recensioni per categoria/hotel", 5, 100, 20)
            
            if st.button("Analizza Soggiorni"):
                with st.spinner("Estrazione durata dai tag e analisi..."):
                    stay_df = analyze_stay_duration(df, min_reviews=min_revs_stay) # da queries.py
                    pdf = stay_df.toPandas() # da Spark a Pandas per visualizzazione
                    
                    if not pdf.empty:
                        # 1. Global Analysis per categoria

                        # Media punteggio per categoria
                        global_stats = pdf.groupby("Stay_Category")['Avg_Score'].mean().reset_index()
                        # Punteggio medio calcolato su tutte le categorie
                        mean_val = global_stats['Avg_Score'].mean()
                        # Calcolo differenza dalla media per evidenziare i piccoli scostamenti
                        global_stats['Delta'] = global_stats['Avg_Score'] - mean_val
                        
                        st.subheader("📊 Performance Globale (Scostamento dalla Media)")
                        st.metric("Punteggio medio di riferimento (globale, calcolato su tutte le categorie)", round(mean_val, 2)) # arrotonda a due cifre decimali
                        st.write("Il delta indica di quanto il punteggio medio di una categoria si discosta dal punteggio medio globale.")
                        st.write("Se una categoria ha un delta positivo, significa che i clienti che hanno fatto soggiorni di quella durata \
                            sono più soddisfatti rispetto alla media generale. Viceversa, un delta negativo indica una soddisfazione inferiore.")

                        # Chart Divergenze
                        base = alt.Chart(global_stats).encode(
                            x=alt.X('Delta:Q', title="Scostamento dalla Media (Punti)", axis=alt.Axis(format='+.3f')),
                            y=alt.Y('Stay_Category:N', sort=["Short Stay (1-3)", "Medium Stay (4-7)", "Long Stay (8+)"], title="Durata Soggiorno"),
                            tooltip=['Stay_Category', 'Avg_Score', 'Delta']
                        )
                        
                        bars = base.mark_bar().encode(
                            color=alt.condition(
                                alt.datum.Delta > 0,
                                alt.value("green"),  # Positivo
                                alt.value("red")     # Negativo
                            )
                        )
                        
                        # Text per valori positivi
                        text_pos = base.transform_filter(
                            alt.datum.Delta > 0
                        ).mark_text(
                            align='left', 
                            dx=5
                        ).encode(
                            text=alt.Text('Avg_Score', format='.3f')
                        )
                        
                        # Text per valori negativi (o zero)
                        text_neg = base.transform_filter(
                            alt.datum.Delta <= 0
                        ).mark_text(
                            align='right', 
                            dx=-5
                        ).encode(
                            text=alt.Text('Avg_Score', format='.3f')
                        )
                        
                        st.altair_chart(bars + text_pos + text_neg, width='stretch')
                        
                        # 2. Top Hotels per Categoria
                        st.write("**🏆 Top Short Stays**")
                        short_df = pdf[pdf['Stay_Category'] == "Short Stay (1-3)"].sort_values("Avg_Score", ascending=False).head(5)
                        st.dataframe(short_df[['Hotel_Name', 'Avg_Score', 'Review_Count']].style.format("{:.2f}", subset=['Avg_Score']), width='stretch')
                        
                        st.write("**🏆 Top Medium Stays**")
                        med_df = pdf[pdf['Stay_Category'] == "Medium Stay (4-7)"].sort_values("Avg_Score", ascending=False).head(5)
                        st.dataframe(med_df[['Hotel_Name', 'Avg_Score', 'Review_Count']].style.format("{:.2f}", subset=['Avg_Score']), width='stretch')
                        
                        st.write("**🏆 Top Long Stays**")
                        long_df = pdf[pdf['Stay_Category'] == "Long Stay (8+)"].sort_values("Avg_Score", ascending=False).head(5)
                        st.dataframe(long_df[['Hotel_Name', 'Avg_Score', 'Review_Count']].style.format("{:.2f}", subset=['Avg_Score']), width='stretch')
                            
                        # 3. Scatter Plot Dettagliato
                        st.subheader("Dettagli Performance Hotel")
                        st.write("Ogni punto è un hotel in una specifica categoria di durata.")
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                        
                        scatter = alt.Chart(pdf).mark_circle(size=60).encode(
                            x=alt.X('Avg_Nights_Actual:Q', title='Media Notti Reali'),
                            y=alt.Y('Avg_Score:Q', scale=alt.Scale(domain=[6, 10]), title='Punteggio Medio'),
                            color='Stay_Category:N',
                            tooltip=['Hotel_Name', 'Stay_Category', 'Avg_Score', 'Review_Count', 'Avg_Nights_Actual']
                        ).interactive()
                        st.altair_chart(scatter, width='stretch')
                        
                    else:
                        st.warning("Nessun dato trovato con i filtri correnti.")

        # Query: Reviewer Experience
        elif query_options[selected_query] == "reviewer_experience":
            st.markdown("""
            #### 🎓 L'Esperienza conta? Novizi vs Esperti
            
            Analizza se il punteggio delle recensioni degli hotel cambia in base all'esperienza dei recensori.
            
            I recensori sono considerati più o meno **esperti** in base al numero di recensioni scritte, venendo classificati nelle seguenti categorie:
            
            *   **Novice (< 5 recensioni)**: novizio, viaggiatore occasionale.
            *   **Intermediate (5-25 recensioni)**: viaggiatore regolare.
            *   **Expert (> 25 recensioni)**: critico o frequent flyer.
            
            Per ogni hotel, viene calcolato l'**Experience Gap (Expert - Novice)**, ovvero la differenza tra il punteggio medio assegnato dagli esperti e quello assegnato dai novizi:
            *   Valore **Positivo**: gli esperti danno voti più alti dei novizi (è un buon segnale, perchè le recensioni dei viaggiatori esperti sono considerate più affidabili).
            *   Valore **Negativo**: gli esperti sono più critici dei novizi (potrebbe essere indice di un hotel che delude le aspettative più elevate).
            """)
            
            min_revs_level = st.slider("Minimo recensioni per livello di esperienza", 5, 100, 10, help="Gli hotel con poche recensioni non sono considerati per evitare distorsioni.")
            
            if st.button("Analizza Esperienza Recensori"):
                with st.spinner("Classificazione recensori e calcolo gap..."):
                    exp_df = analyze_reviewer_experience(df, min_reviews_per_level=min_revs_level)
                    exp_pdf = exp_df.toPandas()
                    
                    if not exp_pdf.empty:
                        # KPI (Key Performance Indicator)
                        avg_gap = exp_pdf['Experience_Gap'].mean()
                        st.metric("Gap Medio (Esperti - Novizi): se negativo, gli esperti sono mediamente più severi dei novizi", f"{avg_gap:+.2f}")
                        
                        st.subheader("🧐 I più criticati dagli Esperti (Gap Negativo)")
                        critics = exp_pdf.sort_values("Experience_Gap").head(10)
                        st.dataframe(critics[['Hotel_Name', 'Novice_Avg_Score', 'Intermediate_Avg_Score', 'Expert_Avg_Score', 'Experience_Gap', 'Total_Analyzed_Reviews', 'Novice_Count', 'Intermediate_Count', 'Expert_Count']].style.format("{:.2f}", subset=['Novice_Avg_Score', 'Intermediate_Avg_Score', 'Expert_Avg_Score', 'Experience_Gap']).background_gradient(subset=['Experience_Gap'], cmap='RdYlGn'), width='stretch')
                        
                        st.subheader("🤝 I più apprezzati dagli Esperti (Gap Positivo)")
                        fans = exp_pdf.sort_values("Experience_Gap", ascending=False).head(10)
                        st.dataframe(fans[['Hotel_Name', 'Novice_Avg_Score', 'Intermediate_Avg_Score', 'Expert_Avg_Score', 'Experience_Gap', 'Total_Analyzed_Reviews', 'Novice_Count', 'Intermediate_Count', 'Expert_Count']].style.format("{:.2f}", subset=['Novice_Avg_Score', 'Intermediate_Avg_Score', 'Expert_Avg_Score', 'Experience_Gap']).background_gradient(subset=['Experience_Gap'], cmap='RdYlGn'), width='stretch')
                        
                        st.subheader("Grafico: Novizi vs Esperti")
                        st.write("Confronto voti: Novizi (X) vs Esperti (Y).")
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                        
                        chart = alt.Chart(exp_pdf).mark_circle(size=60).encode(
                            x=alt.X('Novice_Avg_Score:Q', title='Voto Novizi', scale=alt.Scale(domain=[5, 10])),
                            y=alt.Y('Expert_Avg_Score:Q', title='Voto Esperti', scale=alt.Scale(domain=[5, 10])),
                            color=alt.Color('Experience_Gap:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0), title='Gap (Expert-Novice)'),
                            tooltip=['Hotel_Name', 'Novice_Avg_Score', 'Expert_Avg_Score', 'Experience_Gap', 'Total_Analyzed_Reviews']
                        ).interactive()
                        
                        line = alt.Chart(pd.DataFrame({'x': [5, 10], 'y': [5, 10]})).mark_line(color='gray', strokeDash=[5, 5]).encode(x='x', y='y')
                        st.altair_chart(chart + line, width='stretch')
                        
                    else:
                        st.warning(f"Nessun hotel trovato con almeno {min_revs_level} recensioni per livello (Novice/Expert).")

    else:
        st.error("Impossibile caricare il dataset. Controlla che 'Hotel_Reviews.csv' sia nella cartella.")

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")
