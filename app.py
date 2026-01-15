import streamlit as st
import pandas as pd
from main import create_spark_session, load_data, get_top_hotels
from ml_model import train_satisfaction_model
from sentiment_analysis import (
    fetch_hotel_reviews, 
    enrich_reviews_with_llm,
    get_negative_topic_frequency,
    get_topic_trends,
    get_topic_score_correlation,
    get_discrepancies
)
import altair as alt

# Configurazione della pagina
st.set_page_config(page_title="Hotel Reviews Analytics", layout="wide")

st.title("📊 Hotel Reviews Analytics con Spark")

# Inizializzazione Spark (Cached per evitare riavvii)
@st.cache_resource
def get_spark_session():
    return create_spark_session("HotelReviewsGUI")

# Caricamento dati (Cached as Resource because Spark DataFrames are not picklable for cache_data)
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
        st.success(f"Dati caricati con successo! Totale righe: {df.count()}")
        
        # Sidebar per controlli
        st.sidebar.header("Opzioni Query")
        
        # Selezione Query
        query_options = {
            "Top Hotels (Avg Score)": "top_hotels",
            "Stima Soddisfazione (ML)": "ml_satisfaction",
            "Sentiment Analysis (Local LLM)": "sentiment_analysis" 
        }
        selected_query = st.sidebar.radio("Scegli la query da eseguire:", list(query_options.keys()))
        
        st.divider()
        st.subheader(f"Query selezionata: {selected_query}")
        
        # Query 1: Top Hotels
        if query_options[selected_query] == "top_hotels":
            num_results = st.number_input("Seleziona il numero di risultati da visualizzare:", min_value=1, max_value=100, value=10)
            if st.button("Trova i migliori hotel"):
                if num_results > 0 and num_results <= 100 and num_results.is_integer():
                    st.write(f"##### Elenco dei migliori {num_results} hotel in base al punteggio medio ottenuto nelle recensioni dei clienti.")
                    with st.spinner("Calcolo risultati in corso..."):
                        result_df = get_top_hotels(df, num_results=num_results)
                        # Converto il dataframe Spark in dataframe Pandas per visualizzare i risultati in Streamlit
                        pandas_df = result_df.toPandas()
                        st.dataframe(pandas_df, use_container_width=True)
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
            
        # Query 2: ML Satisfaction
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
            
        # Query 3: Sentiment Analysis
        elif query_options[selected_query] == "sentiment_analysis":
            st.info("Questa funzionalità richiede **Ollama** installato in locale.")
            
            # --- Configurazione ---
            col1, col2 = st.columns(2)
            with col1:
                hotel_name = st.text_input("Inserisci nome Hotel", value="Ritz Paris")
                model_name = st.text_input("Modello Ollama", value="llama3")
            with col2:
                # Limitiamo per evitare attese infinite in demo
                max_reviews = st.number_input("Max recensioni da analizzare", value=30, max_value=200)

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
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Parole Chiave Negative", 
                    "Trend Temporali", 
                    "Correlazione Voto", 
                    "Dissonanze"
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
                            st.altair_chart(c, use_container_width=True)
                            st.dataframe(impact_df)
                        else:
                            st.info("Nessun dato significativo.")

                with tab4:
                    st.write("### Dissonanze (Voto Alto ma Sentiment Negativo)")
                    threshold = st.slider("Soglia Voto Alto", 7.0, 10.0, 8.0)
                    if st.button("Esegui Query: Dissonanze"):
                        disc_df = get_discrepancies(enriched_df, score_threshold=threshold)
                        if not disc_df.empty:
                            st.warning(f"Trovate {len(disc_df)} recensioni sospette.")
                            for i, row in disc_df.iterrows():
                                st.write(f"**Voto: {row['Reviewer_Score']}** - {row['Review_Text']}")
                                st.divider()
                        else:
                            st.success("Nessuna dissonanza trovata.")

            else:
                st.info("👆 Clicca su 'Prepara Dati' per iniziare.")
    else:
        st.error("Impossibile caricare il dataset. Controlla che 'Hotel_Reviews.csv' sia nella cartella.")

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")
