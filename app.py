import streamlit as st
import pandas as pd
from main import create_spark_session, load_data, get_top_hotels
from queries import get_top_hotels_by_nation, analyze_review_trends
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
            "Top Hotels (Avg Score)": "top_hotels",
            "Stima Soddisfazione (ML)": "ml_satisfaction",
            "Sentiment Analysis (Local LLM)": "sentiment_analysis"
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
            st.write("Questa analisi calcola il **trend temporale** dei punteggi per ogni hotel utilizzando la **Regressione Lineare**.")
            st.write("Viene identificato se la reputazione dell'hotel è in crescita (📈), decrescita (📉) o stabile (➖).")
            
            if st.button("Calcola Trend per tutti gli Hotel"):
                with st.spinner("Calcolo regressione lineare per ogni hotel in corso... (potrebbe richiedere qualche secondo)"):
                    trends_df = analyze_review_trends(df)
                    # Convert to Pandas for display
                    trends_pdf = trends_df.toPandas()
                    
                    if not trends_pdf.empty:
                        # Top Improving
                        st.subheader("🏆 Top 10 Hotel in Crescita")
                        improving = trends_pdf[trends_pdf['Trend_Slope'] > 0].sort_values('Trend_Slope', ascending=False).head(10)
                        st.dataframe(improving[['Hotel_Name', 'Trend_Slope', 'Review_Count', 'Trend_Description', 'Average_Score_Calculated']], use_container_width=True)
                        
                        # Top Declining
                        st.subheader("⚠️ Top 10 Hotel in Calo")
                        declining = trends_pdf[trends_pdf['Trend_Slope'] < 0].sort_values('Trend_Slope', ascending=True).head(10)
                        st.dataframe(declining[['Hotel_Name', 'Trend_Slope', 'Review_Count', 'Trend_Description', 'Average_Score_Calculated']], use_container_width=True)
                        
                        # Scatter Plot of Slopes vs Average Score
                        st.subheader("Distribuzione Trend vs Punteggio Medio")
                        chart = alt.Chart(trends_pdf).mark_circle(size=60).encode(
                            x=alt.X('Average_Score_Calculated', title='Punteggio Medio Calcolato'),
                            y=alt.Y('Trend_Slope', title='Trend (Slope)'),
                            color='Trend_Description',
                            tooltip=['Hotel_Name', 'Trend_Slope', 'Review_Count']
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                        
                    else:
                        st.warning("Nessun trend calcolato. Verifica i dati.")

        # Query 1: Top Hotels
        elif query_options[selected_query] == "top_hotels":
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
                            st.altair_chart(c, use_container_width=True)
                            st.dataframe(impact_df)
                        else:
                            st.info("Nessun dato significativo.")



            else:
                st.info("👆 Clicca su 'Prepara Dati' per iniziare.")
    else:
        st.error("Impossibile caricare il dataset. Controlla che 'Hotel_Reviews.csv' sia nella cartella.")

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")
