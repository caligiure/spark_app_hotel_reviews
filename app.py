import sys
import os

# Fix per "Python worker failed to connect back"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import streamlit as st
import pandas as pd
import altair as alt
from main import create_spark_session, load_data, get_top_hotels
from queries import get_top_hotels_by_nation, analyze_review_trends, analyze_tag_influence
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
# Inizializzazione Spark (Cached per evitare riavvii)
@st.cache_resource
def get_spark_session():
    return create_spark_session("HotelReviewsGUI")
# Caricamento dati (Cached)
@st.cache_resource
def get_data(_spark):
    return load_data(_spark)

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
            "Migliori Hotel per Nazione": "top_hotels_by_nation",
            "Trend Recensioni (Time Series)": "review_trends",
            "Analisi Influenza Tag (MapReduce)": "tag_influence",
            "Top Hotels (Avg Score)": "top_hotels",
            "Stima Soddisfazione (ML)": "ml_satisfaction",
            "Sentiment Analysis (Local LLM)": "sentiment_analysis",
        }
        selected_query = st.sidebar.radio("Scegli la query da eseguire:", list(query_options.keys()))
        st.divider()
        st.subheader(f"Query selezionata: {selected_query}")
        
        # Query: Top Hotels by Nation
        if query_options[selected_query] == "top_hotels_by_nation":
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

        # Query: Trend Recensioni
        elif query_options[selected_query] == "review_trends":
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
            ### Analisi Influenza dei Tag
            Questa query utilizza un approccio **MapReduce** per analizzare l'impatto dei tag, determinando quali tag ("Couple", "Leisure trip", ecc.) sono associati a voti più alti o più bassi.
            
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
                        
                        chart = alt.Chart(tag_pdf).mark_circle(size=60).encode(
                            x=alt.X('Impact:Q', title='Impatto (Voto Tag - Media Globale)'),
                            y=alt.Y('Reliability_Index:Q', title='Indice di Affidabilità'),
                            color=alt.Color('Average_Score:Q', scale=alt.Scale(scheme='viridis'), title='Voto Medio'),
                            tooltip=['Single_Tag', 'Average_Score', 'Count', 'Impact', 'Reliability_Index', 'Weighted_Impact', 'StdDev_Score']
                        ).interactive()
                        st.altair_chart(chart, width='stretch')
                        
                    else:
                        st.warning("Nessun tag trovato con i filtri selezionati.")

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
