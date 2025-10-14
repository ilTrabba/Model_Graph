# Model Heritage - MVP

Un sistema di Model Lake per il versionamento automatico di modelli di Machine Learning basato sull'analisi dei pesi, con interfaccia web per navigazione e inserimento modelli.

## 🎯 Obiettivo

Model Heritage identifica automaticamente le relazioni genitore-figlio tra modelli ML analizzando esclusivamente i loro pesi, senza fare affidamento su metadati o convenzioni di naming. Il sistema raggruppa i modelli in famiglie strutturalmente simili e ricostruisce la genealogia utilizzando l'algoritmo MoTHer.

## 🏗️ Architettura MVP

### Backend (Flask)
- **API REST** per upload, listing e dettaglio modelli
- **Database SQLite** con tabelle Model e Family
- **Algoritmo stub** per versionamento automatico:
  - Estrazione signature architettonica dai pesi
  - Clustering in famiglie per similarità strutturale
  - Ricerca parent-child intra-famiglia
- **CORS** abilitato per integrazione frontend

### Frontend (React)
- **Navbar** con "Model Heritage", "Models" e "Add Model"
- **Homepage** con descrizione del sistema e funzionalità
- **Catalogo modelli** con ricerca alfabetica e filtri
- **Form upload** accessibile via navbar
- **Dettaglio modello** con lineage e metadati
- **Design responsive** con Tailwind CSS e shadcn/ui

## 🚀 Avvio Rapido

### Prerequisiti
- Python 3.11+
- Node.js 20+
- pnpm

### Backend
```bash
cd model_heritage_backend
source venv/bin/activate
python run_server.py
```
Il backend sarà disponibile su `http://localhost:5001`

### Frontend
```bash
cd model_heritage_frontend
pnpm run dev --host
```
Il frontend sarà disponibile su `http://localhost:5173`

## 📁 Struttura del Progetto

```
model_heritage_backend/
├── src/
│   ├── models/
│   │   ├── user.py          # Database base
│   │   └── model.py         # Modelli Model e Family
│   ├── routes/
│   │   ├── user.py          # Route utenti (template)
│   │   └── models.py        # API modelli
│   ├── main.py              # App Flask principale
│   └── database/            # Database SQLite
├── uploads/                 # File modelli caricati
├── venv/                    # Virtual environment
└── run_server.py           # Script avvio server

model_heritage_frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # Componenti shadcn/ui
│   │   └── Navbar.jsx       # Barra di navigazione
│   ├── pages/
│   │   ├── HomePage.jsx     # Homepage
│   │   ├── ModelsPage.jsx   # Catalogo modelli
│   │   ├── ModelDetailPage.jsx # Dettaglio modello
│   │   └── AddModelPage.jsx # Form upload
│   └── App.jsx              # App principale
└── package.json
```

## 🔌 API Endpoints

### Modelli
- `GET /api/models` - Lista tutti i modelli con ricerca opzionale
- `GET /api/models/{id}` - Dettaglio modello con lineage
- `POST /api/models` - Upload nuovo modello
- `GET /api/families` - Lista famiglie di modelli
- `GET /api/families/{id}/models` - Modelli in una famiglia
- `GET /api/stats` - Statistiche sistema

### Esempio Upload
```bash
curl -X POST http://localhost:5001/api/models \
  -F "file=@model.pt" \
  -F "name=My Model" \
  -F "description=Test model"
```

## 🧬 Algoritmo di Versionamento (Stub)

### 1. Estrazione Signature
```python
def extract_weight_signature_stub(file_path):
    # Analizza dimensioni file per stimare parametri
    # Calcola hash strutturale
    # Estrae conteggio layer stimato
    return signature
```

### 2. Assegnazione Famiglia
```python
def assign_to_family_stub(signature):
    # Raggruppa per range di parametri:
    # - small_models: < 1M parametri
    # - medium_models: 1M-100M parametri  
    # - large_models: > 100M parametri
    return family_id
```

### 3. Ricerca Parent
```python
def find_parent_stub(model, family_id):
    # Trova modello più simile per conteggio parametri
    # Calcola confidence basata su similarità
    return parent_id, confidence
```

## 🎨 Interfaccia Utente

### Navbar
- **Logo e titolo**: "Model Heritage" con icona database
- **Sezione Models**: Link al catalogo modelli
- **Pulsante Add Model**: Accesso diretto al form di upload

### Catalogo Modelli
- **Ricerca**: Barra di ricerca case-insensitive per nome
- **Card compatte**: Nome, parametri, layer, famiglia, stato
- **Ordinamento**: Alfabetico per nome
- **Stato vuoto**: Messaggio e link per primo upload

### Form Upload
- **File picker**: Drag & drop per .safetensors, .pt, .bin, .pth
- **Metadati**: Nome (obbligatorio) e descrizione (opzionale)
- **Validazione**: Controllo formato e dimensioni
- **Feedback**: Progress e messaggi di errore/successo

### Dettaglio Modello
- **Specifiche**: Parametri, layer, hash strutturale
- **Lineage**: Parent e children con confidence score
- **Famiglia**: Badge e informazioni gruppo
- **Timeline**: Date creazione e processamento

## 🔧 Configurazione

### Backend
- **Porta**: 5001 (evita conflitti con altri servizi)
- **Database**: SQLite in `src/database/app.db`
- **Upload**: Directory `uploads/` con controllo checksum
- **CORS**: Abilitato per tutti i domini

### Frontend  
- **Porta**: 5173 (default Vite)
- **API URL**: Hardcoded `http://localhost:5001`
- **Routing**: React Router per SPA
- **Styling**: Tailwind CSS + shadcn/ui

## 🧪 Test Funzionali

### Test Upload
1. Accedi a `http://localhost:5173`
2. Clicca "Add Model" nella navbar
3. Carica un file .pt di test
4. Compila nome e descrizione
5. Verifica upload e processamento

### Test Catalogo
1. Vai su "Models" nella navbar
2. Verifica visualizzazione modelli
3. Testa ricerca per nome
4. Clicca su un modello per dettagli

### Test API
```bash
# Lista modelli
curl http://localhost:5001/api/models

# Statistiche
curl http://localhost:5001/api/stats

# Upload test
curl -X POST http://localhost:5001/api/models \
  -F "file=@test_model.pt" \
  -F "name=Test Model"
```

## 🚧 Prossimi Sviluppi

### Integrazione MoTHer Reale
- Sostituire stub con algoritmo MoTHer completo
- Implementare analisi pesi PyTorch/SafeTensors
- Aggiungere clustering 4-step con centroidi
- Integrare minimum spanning arborescence

### Scalabilità
- Migrare da SQLite a PostgreSQL
- Aggiungere Redis per caching
- Implementare job queue con RQ/Celery
- Containerizzare con Docker Compose

### UI Avanzate
- Visualizzazione grafo famiglie con vis-network
- Dashboard statistiche con Recharts
- Filtri avanzati (famiglia, parametri, data)
- Export lineage in formati standard

### Produzione
- Autenticazione e autorizzazione
- Rate limiting e validazione robusta
- Monitoring e logging strutturato
- Backup automatico database

## 📝 Note Tecniche

### Limitazioni MVP
- **Algoritmo semplificato**: Stub basato su dimensioni file
- **Storage locale**: File e database non distribuiti  
- **Sicurezza minima**: Nessuna autenticazione
- **Scalabilità limitata**: Single-threaded processing

### Formati Supportati
- **.safetensors**: Formato preferito (sicuro)
- **.pt/.pth**: PyTorch standard
- **.bin**: Pickle binario (legacy)

### Performance
- **Upload**: Limitato a file < 5GB (configurabile)
- **Processing**: Sincrono per MVP (asincrono in produzione)
- **Database**: SQLite adeguato per < 1000 modelli

## 🤝 Contributi

Per estendere il sistema:

1. **Backend**: Aggiungere endpoint in `src/routes/models.py`
2. **Frontend**: Creare componenti in `src/components/`
3. **Database**: Modificare modelli in `src/models/model.py`
4. **Algoritmi**: Implementare in moduli separati

### Convenzioni
- **API**: REST con JSON response
- **Frontend**: Functional components con hooks
- **Styling**: Tailwind classes, evitare CSS custom
- **Database**: SQLAlchemy ORM con migrations

## 📄 Licenza

Progetto sviluppato per scopi educativi e di ricerca.

