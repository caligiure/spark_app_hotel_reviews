import streamlit as st
import pandas as pd
from main import create_spark_session, load_data, get_top_hotels
from ml_model import train_satisfaction_model
from sentiment_analysis import analyze_sentiment_sample
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
        
        with st.spinner("Caricamento risultati in corso..."):
            # Query 1: Top Hotels
            if query_options[selected_query] == "top_hotels":
                num_results = st.number_input("Seleziona il numero di risultati da visualizzare:", min_value=1, max_value=100, value=10)
                if st.button("Trova i migliori hotel"):
                    if num_results > 0 and num_results <= 100 and num_results.is_integer():
                        st.info(f"Elenco dei migliori hotel in base al punteggio medio ottenuto nelle recensioni dei clienti.")
                        result_df = get_top_hotels(df, num_results=num_results)
                        # Converto il dataframe Spark in dataframe Pandas per visualizzare i risultati in Streamlit
                        pandas_df = result_df.toPandas()
                        st.dataframe(pandas_df, use_container_width=True)
                        # Visualizzazione con Altair
                        st.write("#### Grafico Interattivo")
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
                
                col1, col2 = st.columns(2)
                with col1:
                    model_name = st.text_input("Nome Modello Ollama", value="llama3", help="Assicurati di aver fatto 'ollama pull <nome>'")
                with col2:
                    sample_size = st.number_input("Numero di recensioni da analizzare", min_value=1, max_value=20, value=3)
                
                if st.button("Avvia Analisi Sentimento"):
                    st.warning("⚠️ L'analisi richiede tempo (diversi secondi per recensione su CPU). Attendi...")
                    with st.spinner(f"Sto chiedendo a {model_name} di leggere le recensioni..."):
                        sentiment_df = analyze_sentiment_sample(df, sample_size=sample_size, model_name=model_name)
                    
                    if not sentiment_df.empty:
                        st.write("### Risultati Analisi")
                        st.dataframe(sentiment_df)
                        
                        # Simple stats
                        st.write("#### Distribuzione Sentimenti")
                        st.bar_chart(sentiment_df["LLM Sentiment"].value_counts())
                    else:
                        st.error("Nessun risultato o errore di connessione con Ollama.")
    else:
        st.error("Impossibile caricare il dataset. Controlla che 'Hotel_Reviews.csv' sia nella cartella.")

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")
