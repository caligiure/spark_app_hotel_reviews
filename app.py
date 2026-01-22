import sys
import os

# Fix per "Python worker failed to connect back"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
from pyspark.sql import SparkSession
import time
from main import get_top_hotels
from queries import get_top_hotels_by_nation, analyze_review_trends, analyze_tag_influence, analyze_nationality_bias, analyze_local_competitiveness, segment_hotels_kmeans, compare_local_vs_tourist_reviews
from ml_model import train_satisfaction_model
from sentiment_analysis import (
    fetch_hotel_reviews, 
    enrich_reviews_with_llm,
    get_negative_topic_frequency,
    get_topic_trends,
    get_topic_score_correlation,
    get_discrepancies
)

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
            "Top Hotels (Avg Score)": "top_hotels",
            "Stima Soddisfazione (ML)": "ml_satisfaction",
            "Sentiment Analysis (Local LLM)": "sentiment_analysis",
        }
        selected_query = st.sidebar.radio("Scegli la query da eseguire:", list(query_options.keys()))
        
        st.subheader(f"Query selezionata: {selected_query}")

        # Query: Trend Recensioni
        if query_options[selected_query] == "review_trends":
            st.markdown("""
            Questa analisi calcola il **trend temporale** dei punteggi per ogni hotel utilizzando la **Regressione Lineare**.
            
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
            if st.button("Calcola Trend per tutti gli Hotel"):
                with st.spinner("Calcolo regressione lineare per ogni hotel in corso... (potrebbe richiedere qualche secondo)"):
                    # Esegue la query (da queries.py)
                    trends_df = analyze_review_trends(df, min_number_of_reviews = 30)
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
            
            1.  **Map**: Esplode la lista dei tag di ogni recensione.
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

        # Query: Top Hotels
        elif query_options[selected_query] == "top_hotels":
            num_results = st.number_input("Seleziona il numero di risultati da visualizzare:", min_value=1, max_value=100, value=10)
            if st.button("Trova i migliori hotel"):
                if num_results > 0 and num_results <= 100 and num_results.is_integer():
                    st.write(f"##### Elenco dei migliori {num_results} hotel in base al punteggio medio ottenuto nelle recensioni dei clienti.")
                    with st.spinner("Calcolo risultati in corso..."):
                        result_df = get_top_hotels(df, num_results=num_results)
                        # Converto il dataframe Spark in dataframe Pandas per visualizzare i risultati in Streamlit
                        pandas_df = result_df.toPandas()
                        st.dataframe(pandas_df, width='stretch')
                        # Visualizzazione con Altair
                        st.write("#### Grafico Hotel per Punteggio Medio")
                        st.info("Nota: cliccando su un punto del grafico è possibile visualizzare le informazioni relative all'hotel.")
                        chart = alt.Chart(pandas_df).mark_bar().encode(
                            x=alt.X('Average_Score', title='Punteggio Medio', scale=alt.Scale(domain=[pandas_df["Average_Score"].min() - 0.5, 10])),
                            y=alt.Y('Hotel_Name', sort='-x', title='Hotel'),
                            color=alt.Color('Average_Score', scale=alt.Scale(scheme='viridis'), legend=None),
                            tooltip=['Hotel_Name', 'Average_Score', 'Review_Count']
                        ).interactive()
                        st.altair_chart(chart, width='stretch')
                else:
                    st.error("Inserisci un numero valido di risultati.")

        # Query: ML Satisfaction
        elif query_options[selected_query] == "ml_satisfaction":
            if st.button("Addestra Modello e Mostra Coefficienti"):
                st.info("Addestramento modello di regressione lineare in corso...")
                # Passiamo il dataframe Spark
                model, rmse, r2 = train_satisfaction_model(df)
                    
                st.write(f"### Risultati Modello")
                st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.4f}")
                st.write(f"**R2 (R-squared):** {r2:.4f}")
                    
                lr_model = model.stages[-1]
                coeffs = lr_model.coefficients
                intercept = lr_model.intercept
                    
                st.write("#### Coefficienti:")
                st.write(f"- Intercetta: {intercept:.4f}")
                    
                feature_cols = [
                    "Average_Score", 
                    "Total_Number_of_Reviews", 
                    "Review_Total_Negative_Word_Counts",
                    "Review_Total_Positive_Word_Counts",
                    "Total_Number_of_Reviews_Reviewer_Has_Given",
                    "Additional_Number_of_Scoring"
                ]
                    
                coeffs_data = {
                    "Feature": feature_cols,
                    "Coefficient": [float(c) for c in coeffs]
                }
                st.table(pd.DataFrame(coeffs_data))
                    
                    
                st.success("Modello addestrato!")

        # Query: Sentiment Analysis
        elif query_options[selected_query] == "sentiment_analysis":
            st.info("Questa funzionalità richiede **Ollama** installato in locale.")
            # --- Configurazione ---
            col1, col2 = st.columns(2)
            with col1:
                hotel_name = st.text_input("Inserisci nome Hotel", value="Ritz Paris")
                model_name = st.selectbox("Modello Ollama", ["qwen2:0.5b", "tinyllama", "qwen2.5:1.5b", "phi3", "llama3"])
            with col2:
                # Limitiamo per evitare attese infinite in demo
                max_reviews = st.number_input("Max recensioni da analizzare", value=30, min_value=1, max_value=200)
            # --- Stato Sessione per Caching ---
            if "enriched_df" not in st.session_state:
                st.session_state["enriched_df"] = None
            if "current_hotel" not in st.session_state:
                st.session_state["current_hotel"] = ""
            # --- Bottone "Prepara Dati" (Esegue LLM) ---
            st.divider()
            if st.button("🚀 1. Prepara/Aggiorna Dati (Esegui LLM)"):
                with st.spinner(f"Scarico recensioni per '{hotel_name}' e avvio analisi con {model_name}..."):
                    # 1. Fetch
                    raw_pandas_df = fetch_hotel_reviews(df, hotel_name, limit=max_reviews)
                    if raw_pandas_df.empty:
                        st.error(f"Nessuna recensione trovata per '{hotel_name}'.")
                    else:
                        st.write(f"Trovate {len(raw_pandas_df)} recensioni. Analisi in corso...")
                        # 2. Enrich
                        enriched = enrich_reviews_with_llm(raw_pandas_df, model_name=model_name)
                        # 3. Store in Session
                        st.session_state["enriched_df"] = enriched
                        st.session_state["current_hotel"] = hotel_name
                        st.success("Analisi completata! Ora puoi eseguire le query qui sotto.")
            
            # --- Sezione Query Separate ---
            if st.session_state["enriched_df"] is not None:
                enriched_df = st.session_state["enriched_df"]
                st.write(f"✅ Dati pronti per: **{st.session_state['current_hotel']}** ({len(enriched_df)} recensioni)")
                
                tab1, tab2, tab3 = st.tabs([
                    "Parole Chiave Negative", 
                    "Trend Temporali", 
                    "Correlazione Voto"
                ])
                
                with tab1:
                    st.write("### Frequenza Problemi (Recensioni Negative)")
                    if st.button("Esegui Query: Parole Chiave"):
                        freq_df = get_negative_topic_frequency(enriched_df)
                        if not freq_df.empty:
                            st.bar_chart(freq_df.set_index("Topic"))
                            st.dataframe(freq_df)
                        else:
                            st.info("Nessun topic negativo trovato.")

                with tab2:
                    st.write("### Trend dei Topic nel Tempo")
                    if st.button("Esegui Query: Trend"):
                        trends_df = get_topic_trends(enriched_df)
                        if not trends_df.empty:
                            st.line_chart(trends_df, x="Month_Year", y="Count", color="LLM_Topics")
                            st.dataframe(trends_df)
                        else:
                            st.info("Dati insufficienti per i trend.")

                with tab3:
                    st.write("### Impatto dei Topic sul Voto")
                    st.write("Quanto un topic alza o abbassa la media voti rispetto alla media dell'hotel?")
                    if st.button("Esegui Query: Correlazione"):
                        impact_df, avg_hotel = get_topic_score_correlation(enriched_df)
                        if not impact_df.empty:
                            st.metric("Media Voto Hotel", f"{avg_hotel:.2f}")
                            
                            # Chart
                            c = alt.Chart(impact_df).mark_bar().encode(
                                x='Impact:Q',
                                y=alt.Y('LLM_Topics:N', sort='x'),
                                color=alt.condition(
                                    alt.datum.Impact > 0,
                                    alt.value("green"),
                                    alt.value("red")
                                )
                            )
                            st.altair_chart(c, width='stretch')
                            st.dataframe(impact_df)
                        else:
                            st.info("Nessun dato significativo.")
            else:
                st.info("👆 Clicca su 'Prepara Dati' per iniziare.")

    else:
        st.error("Impossibile caricare il dataset. Controlla che 'Hotel_Reviews.csv' sia nella cartella.")

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")
