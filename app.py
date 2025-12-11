import streamlit as st
import pandas as pd
from main import create_spark_session, load_data, get_top_hotels

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
            "Top 10 Hotel (Avg Score)": "top_hotels",
            # "Altra query futura": "other" 
        }
        selected_query = st.sidebar.selectbox("Scegli la query da eseguire:", list(query_options.keys()))
        
        if st.sidebar.button("Esegui Query"):
            st.subheader(f"Risultati: {selected_query}")
            
            with st.spinner("Esecuzione query Spark..."):
                if query_options[selected_query] == "top_hotels":
                    result_df = get_top_hotels(df)
                    
                    # Convert to Pandas for Streamlit display
                    # Note: limit(10) is already applied in get_top_hotels, so it's safe.
                    # For larger results, we should apply limit here.
                    pandas_df = result_df.toPandas()
                    
                    st.dataframe(pandas_df, use_container_width=True)
                    
                    # Optional: Charts
                    st.bar_chart(pandas_df.set_index("Hotel_Name")["Average_Score"])
    else:
        st.error("Impossibile caricare il dataset. Controlla che 'Hotel_Reviews.csv' sia nella cartella.")

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")
