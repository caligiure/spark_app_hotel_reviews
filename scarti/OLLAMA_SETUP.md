# Configurazione Ollama per Sentiment Analysis

Per eseguire la Sentiment Analysis in locale, utilizzeremo **Ollama**.
Segui questi passaggi per installarlo e scaricare il modello necessario.

## 1. Installazione

### Windows
1.  Scarica l'installer ufficiale da [ollama.com/download/windows](https://ollama.com/download/windows).
2.  Esegui il file `.exe` scaricato.
3.  Segui le istruzioni a schermo per completare l'installazione.

### Mac / Linux
- **Mac**: Scarica da [ollama.com/download/mac](https://ollama.com/download/mac).
- **Linux**: Esegui nel terminale: `curl -fsSL https://ollama.com/install.sh | sh`

## 2. Verifica Installazione
Apri il terminale (PowerShell su Windows) ed esegui:
```powershell
ollama --version
```
Se vedi la versione (es. `ollama version 0.1.30`), è installato correttamente.

## 3. Scaricare il Modello
Per la sentiment analysis useremo `llaama3` (o `mistral`), modelli leggeri e performanti.
Apri il terminale ed esegui i seguenti comandi:

**Opzione A: Llama 3 (Consigliato, ~4.7GB)**
```powershell
ollama pull llama3
```

**Opzione B: Mistral (Alternativa leggera, ~4.1GB)**
```powershell
ollama pull mistral
```

**Opzione C: Modelli Ultra-Veloci (Consigliati per CPU)**
Se `phi3` è troppo lento, prova questi modelli molto più piccoli:

*   **TinyLlama** (1.1GB - Molto veloce):
    ```powershell
    ollama pull tinyllama
    ```
*   **Qwen2 0.5B** (350MB - Istantaneo, ma meno preciso):
    ```powershell
    ollama pull qwen2:0.5b
    ```

## 4. Test Rapido
Per verificare che il modello funzioni, scrivi:
```powershell
ollama run llama3 "This hotel was amazing! I loved the breakfast. Sentiment?"
```
Se risponde correttamente, sei pronto.

> [NOTE]
> Il server API di Ollama parte automaticamente in background sulla porta **11434**. Il nostro script Python si collegherà a `http://localhost:11434`.
