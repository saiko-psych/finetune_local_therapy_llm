# Finetuning eines Large Language Models für therapeutische Anwendungen

## 📋 Projektübersicht

Dieses Projekt untersucht die Wirksamkeit des Finetunings von Large Language Models (LLMs) für therapeutische Gespräche durch den Vergleich eines Original Gemma-3-4B Modells mit einer selbst trainierten Version auf realen Mental Health Counseling Daten.

## 🎯 Forschungsziel

**Hauptfragestellung:** Kann durch Finetuning die Qualität therapeutischer Antworten eines LLMs signifikant verbessert werden - und das mit einem Modell, das datenschutzkonform lokal auf Consumer-Hardware läuft?

**Motivation für lokale Modelle:**
- **Datenschutz:** Sensible therapeutische Gespräche bleiben vollständig lokal
- **Verfügbarkeit:** Keine Abhängigkeit von Cloud-APIs oder Internetverbindung
- **Kosten:** Keine laufenden API-Kosten nach einmaligem Setup
- **Kontrolle:** Vollständige Kontrolle über Modellverhalten und Updates
- **Zugänglichkeit:** Demokratisierung von KI-Tools für therapeutische Anwendungen

**Hypothese:** Ein auf therapeutischen Gesprächsdaten finegetuntes 4B-Parameter Modell kann lokal auf Standard-Hardware (8-16GB GPU) betrieben werden und dabei höhere semantische Ähnlichkeit zu echten Therapeutenantworten zeigen als das ursprüngliche Basismodell.

## 🔬 Methodologie

### 1. Datengrundlage

**Primärdataset (Finetuning):**
- `entfane/psychotherapy` - Strukturierte therapeutische Gespräche mit Metadaten
- Enthält Informationen zu Krankheitsbild und Therapiestadium
- Aufbereitung: Extraktion von Kontext-Response-Paaren mit Metainformationen

**Evaluationsdataset:**
- `Amod/mental_health_counseling_conversations` - 3.512 echte Therapeut-Patient Gespräche
- Zwei Testbedingungen:
  - **Gefiltert:** 50-1000 Zeichen (Context), 50-2000 Zeichen (Response)
  - **Ungefiltert:** Zufällige Auswahl aus allen Daten
- Sample-Größe: 200 pro Bedingung = 400 total

### 2. Technische Implementierung

#### Modell-Setup: Lokale Deployment-Strategie
```python
# Basismodell: Google Gemma-3-4B-IT
# Bewusste Wahl eines kleineren Modells für lokale Nutzung
model_id = "google/gemma-3-4b-it"

# 4-Bit Quantisierung für Consumer-Hardware Kompatibilität
BitsAndBytesConfig:
- 4-bit Quantisierung (NF4)     # Reduziert VRAM von ~16GB auf ~4GB
- Float16 Compute Dtype         # Weitere Memory-Optimierung
- CPU Offloading aktiviert      # Nutzt System-RAM als Backup
- Double Quantisation           # Zusätzliche Kompression
```

**Warum Gemma-3-4B statt größerer Modelle?**
- **Hardware-Anforderungen:** Läuft auf 8GB GPU (RTX 3070/4060 Ti)
- **Inference-Geschwindigkeit:** ~50-100 Tokens/Sekunde auf Consumer-Hardware
- **Speicherbedarf:** ~4GB VRAM quantisiert vs. ~60GB+ für 70B Modelle
- **Energieeffizienz:** Lokal betreibbar ohne Serverfarm

#### LoRA Konfiguration
```python
LoraConfig:
- r=16 (Rank)
- lora_alpha=32
- target_modules=["q_proj", "v_proj"]
- lora_dropout=0.05
- task_type="CAUSAL_LM"
```

#### Training-Parameter
```python
TrainingArguments:
- Batch Size: 1 (mit Gradient Accumulation)
- Learning Rate: 1e-4
- Epochs: 10
- Mixed Precision: FP16
- LR Scheduler: Linear mit Warmup
- Early Stopping verfügbar
```

### 3. Datenaufbereitung

#### Kontext-Extraktion mit Metadaten
Die Trainingsdaten werden strukturiert aufbereitet:
- **Metainformationen:** `[Illness: {illness}] [Stage: {stage}]`
- **Kontextaufbau:** Akkumulation der Gesprächshistorie
- **Pairs-Extraktion:** Kontext → Therapeut-Response Paare

#### Tokenisierung
- Max Length: 1024 Token
- Padding: "max_length"
- Truncation: True
- Labels = Input IDs (Causal Language Modeling)

### 4. Experimentelles Design

**Evaluationsprozess:**
1. **Input:** Patient Context aus Evaluationsdataset
2. **Generation:** 
   - Original Model → Response A
   - Finetuned Model → Response B
3. **Ground Truth:** Echte Therapeut-Antwort
4. **Vergleich:** Cosine Similarity (TF-IDF basiert)

**Metriken:**
- Original vs Ground Truth
- Finetuned vs Ground Truth  
- Original vs Finetuned

### 5. Robustheit & Reproduzierbarkeit

**Fehlerbehandlung:**
- Checkpoint System (alle 10 Samples)
- 3 Retry-Versuche bei API-Fehlern
- Automatische Crash Recovery
- Vollständiges Logging

**Datenintegrität:**
- Deterministische Seeds (42)
- Versionskontrolle aller Parameter
- Reproduzierbare Train/Test Splits

## 📊 Evaluationsstrategie

### Quantitative Metriken

**Cosine Similarity Analysis:**
- **Basis:** TF-IDF Vektorisierung
- **Vergleiche:** 3 pro Sample (1200 total)
- **Aggregation:** Mittelwerte, Standardabweichungen

**Statistische Tests:**
- Verbesserungsraten (Anteil Samples mit Similarity-Gain > 0)
- Effektgrößen zwischen gefilterten/ungefilterten Bedingungen
- Signifikanztests für Modellunterschiede

### Visualisierung
- Verteilungsplots der Similarity-Scores
- Vergleichsgrafiken Original vs Finetuned
- Scatter-Plots für Korrelationsanalyse
- Box-Plots für Variabilitätsvergleiche

## 🛠️ Technische Anforderungen - Consumer-Hardware Ready

### Minimum Hardware (Budget-Setup)
- **GPU:** RTX 3060 12GB / RTX 4060 Ti 16GB / RX 6700 XT 12GB
- **RAM:** 16GB System-RAM (8GB für Modell-Offloading)
- **Storage:** 15GB freier Speicher (Modell + Checkpoints)
- **Geschätzte Kosten:** ~400-600€ gebrauchte GPU

### Empfohlene Hardware (Komfort-Setup)  
- **GPU:** RTX 3070/4070 (8GB+) oder RTX 3080/4080 (10GB+)
- **RAM:** 32GB System-RAM (für größere Batch-Sizes)
- **Storage:** SSD mit 50GB+ freiem Speicher
- **Geschätzte Kosten:** ~600-1000€ gebrauchte GPU

### Performance-Vergleich lokaler Betrieb
```
Inference-Geschwindigkeit (Consumer-Hardware):
├── RTX 3060 12GB:  ~30-50 Tokens/Sekunde
├── RTX 3070 8GB:   ~40-70 Tokens/Sekunde  
├── RTX 4070 12GB:  ~60-100 Tokens/Sekunde
└── RTX 4080 16GB:  ~80-120 Tokens/Sekunde

Vergleich Cloud-APIs:
├── GPT-4: ~20-40 Tokens/Sekunde (+ Latenz + Kosten)
├── Claude: ~15-30 Tokens/Sekunde (+ Latenz + Kosten)
└── Lokales 4B: ~50-100 Tokens/Sekunde (keine Latenz, keine Kosten)
```

### Datenschutz-Vorteile lokaler Modelle
- **Offline-Betrieb:** Keine Internetverbindung für Inference nötig
- **Zero Data Transmission:** Gespräche verlassen niemals den lokalen Rechner
- **DSGVO-Konform:** Keine Übertragung personenbezogener Daten an Dritte
- **Audit-Fähigkeit:** Vollständige Kontrolle über Datenverarbeitung
- **Therapeutische Schweigepflicht:** Höchster Datenschutz für sensible Inhalte

### Software Dependencies
```python
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
bitsandbytes>=0.41.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 📁 Projektstruktur

```
therapy-llm-finetuning/
├── data/
│   ├── processed/          # Aufbereitete Datasets
│   └── results/           # Evaluationsergebnisse
├── src/
│   ├── finetune_gemma.py  # Haupttraining-Skript
│   ├── evaluation.py      # Evaluations-Pipeline
│   └── utils/             # Hilfsfunktionen
├── models/
│   └── finetuned-gemma/   # Trainierte Modell-Checkpoints
├── logs/                  # Training & Evaluation Logs
├── notebooks/             # Jupyter Notebooks für Analyse
├── results/               # Plots und Reports
├── requirements.txt       # Python Dependencies
└── README.md             # Projektbeschreibung
```

## 🔍 Wissenschaftlicher Beitrag

### Methodische Innovation
- **Quantitative LLM-Evaluation:** Objektive Messung der Finetuning-Effektivität auf therapeutischen Daten
- **Robuste Experimentgestaltung:** Systematischer Vergleich mit echten Baseline-Antworten
- **Lokale AI-Architektur:** Demonstration datenschutzkonformer KI-Systeme für sensible Anwendungen
- **Consumer-Hardware Optimierung:** Beweis der Machbarkeit komplexer AI-Tasks auf Standard-PCs

### Praktische Relevanz für lokale KI-Systeme
- **Datenschutz-Revolution:** Therapeutische KI ohne Cloud-Abhängigkeit
- **Demokratisierung:** KI-Tools zugänglich für jeden mit Standard-Gaming-PC
- **Kosteneffizienz:** Einmalige Anschaffung vs. laufende API-Kosten (GPT-4: ~$0.03/1K Tokens)
- **Verfügbarkeit:** 24/7 Betrieb ohne Internetabhängigkeit oder Service-Ausfälle
- **Anpassbarkeit:** Vollständige Kontrolle über Modellverhalten und ethische Richtlinien

### Gesellschaftlicher Impact
- **Therapeutische Versorgung:** Unterstützung in unterversorgten Gebieten
- **Kostenreduktion:** Senkung der Barrieren für mental health support
- **Forschungsförderung:** Open-Source Ansatz für weitere wissenschaftliche Entwicklung
- **Ethik-Standard:** Referenzimplementierung für verantwortliche lokale KI

## 💡 Warum lokale Modelle statt Cloud-APIs?

### Datenschutz & Sicherheit
| Aspekt | Cloud-APIs (GPT-4, Claude) | Lokales 4B Modell |
|--------|----------------------------|-------------------|
| **Datenübertragung** | Alle Inputs → Cloud Server | Keine (100% lokal) |
| **Datenspeicherung** | Unbekannt/Temporär | Vollständige Kontrolle |
| **Compliance** | Abhängig vom Anbieter | DSGVO/HIPAA ready |
| **Auditierbarkeit** | Schwarz-Box | Open Source |
| **Therapeutische Schweigepflicht** | Rechtsunsicherheit | Garantiert erfüllt |



### Verfügbarkeit & Kontrolle
- **24/7 Betrieb:** Keine API-Limits oder Service-Ausfälle
- **Offline-Fähigkeit:** Funktioniert ohne Internetverbindung
- **Latenz:** Sub-Sekunde Response vs. Cloud-Latenz
- **Anpassbarkeit:** Vollständige Kontrolle über Modellverhalten
- **Updates:** Selbstbestimmte Modell-Updates ohne Qualitätsverlust


## 🚀 Ausführung

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Finetuning
```bash
python src/finetune_gemma.py
```

### 3. Evaluation
```bash
python src/evaluation.py
```

### 4. Analyse
```bash
jupyter notebook notebooks/analysis.ipynb
```

## 📝 Limitationen & lokale Modell-Realitäten

### Technische Limitationen
- **Modellgröße:** 4B Parameter vs. 175B+ bei GPT-4 (erwartete Qualitätsdifferenz)
- **Kontextlänge:** 8K Token Limit vs. 128K bei neueren Cloud-Modellen
- **Multimodalität:** Text-only vs. Vision+Audio Fähigkeiten größerer Modelle
- **Training-Daten:** Begrenzte lokale Trainingsdaten (500 Samples) vs. Cloud-Training

### Evaluation-Limitationen  
- **Evaluationsmetrik:** Cosine Similarity erfasst nicht alle Aspekte therapeutischer Qualität
- **Subjektivität:** Keine menschliche Bewertung der generierten Antworten
- **Sprache:** Evaluierung nur auf englischsprachigen Daten
- **Kontext:** Evaluation nur auf Einzelturn-Gesprächen

### Warum diese Limitationen akzeptabel sind
- **Privacy-First:** Datenschutz wiegt Qualitätsverluste auf
- **Good-Enough:** 80% Qualität bei 100% Privacy oft ausreichend
- **Verbesserbar:** Lokale Modelle entwickeln sich schnell weiter
- **Spezialisierung:** Finetuning kann Domain-spezifische Defizite ausgleichen

## 🔮 Zukünftige Erweiterungen & lokale AI-Roadmap

### Kurzfristig (3-6 Monate)
- **Human Evaluation:** Bewertung durch lizenzierte Therapeuten
- **Multilinguale Evaluation:** Test auf deutsch- und anderssprachigen Daten
- **Hardware-Optimierung:** Benchmarks auf verschiedenen Consumer-GPUs
- **Deployment-Tools:** Docker Container für einfache Installation

### Mittelfristig (6-12 Monate)
- **Längere Dialoge:** Multi-Turn Gespräche und Session-Memory
- **Lokale RAG-Integration:** Verbindung mit lokalen Wissensdatenbanken
- **Mobile Deployment:** Optimierung für Smartphones/Tablets
- **Federated Learning:** Dezentrale Modellverbesserung ohne Datenteilung

### Langfristig (1-2 Jahre)
- **Lokale Multimodalität:** Integration von Speech-to-Text/Text-to-Speech
- **Edge-Computing:** Deployment auf Raspberry Pi / Edge-Devices
- **Community-Training:** Crowd-sourced Verbesserung lokaler Therapie-Modelle
- **Regulatorische Frameworks:** Standards für lokale medizinische KI-Systeme

### Vision: Demokratisierte therapeutische KI
*"Jeder Mensch sollte Zugang zu hochwertiger, datenschutzkonformer KI-Unterstützung für mental health haben - unabhängig von Internet, Budget oder geografischer Lage."*

---

**Autor:** [David Matischek]  
**Institution:** [Karl Franzens Universität Graz]  
**Datum:** Juli 2025  
**Kontakt:** [david.matischek@edu.uni-graz.at]
