# Finetuning eines Large Language Models für therapeutische Anwendungen

## 📋 Projektübersicht


Dieses Projekt untersucht die Wirksamkeit des Finetunings von Large Language Models (LLMs) für therapeutische Gespräche durch den Vergleich eines Original Gemma-3-4B Modells mit einer selbst trainierten Version auf realen Mental Health Counseling Daten. Dieses Projekt baut auf diesen Projekt auf: [GitHub](https://github.com/MGerschuetz/localTherapy) 

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
- Sample-Größe: 6 (Der Rechenaufwand war sonst zu grpß, bzw ist es sich zeitlich nicht mehr ausgegangen)

### 2. Technische Implementierung


**Warum Gemma-3-4B statt größerer Modelle?**
- **Hardware-Anforderungen:** Läuft auf 8GB GPU (RTX 3070/4060 Ti)
- **Inference-Geschwindigkeit:** ~50-100 Tokens/Sekunde auf Consumer-Hardware
- **Speicherbedarf:** ~4GB VRAM quantisiert vs. ~60GB+ für 70B Modelle
- **Energieeffizienz:** Lokal betreibbar ohne Serverfarm



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
- Checkpoint System (alle 2 Samples (das kann geändert werden))
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



## 🚀 Ausführung

### 1. Environment Setup

-  virtuelles environment in pyhton 3.10 anlegen und benötigte pakete installieren!

### 2. Finetuning
environment aktiviern und im Terminal dies ausführen

```bash
start /B python finetune_gemma.py > finetune_gemma.log 2>&1
powershell -Command "Get-Content .\finetune_gemma.log -Wait -Tail 20"
```

### 3. Evaluation
```bash
python model_evaluation.py
```

### 4. Vergleich und Analyse

Dateien und Visualisierungen werden automatisch erstellt! und sind oben im file evaluation_result6samples zu finden!



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

## 🔮 Erkenntnisse, Fazit und Ausblick

es zeigte sich bei diesem Test, dass das finetuned Modell mehr Ähnlichkeit zu der grount truth zeigte als das originale Modell. Doch habe ich dies vorerst nur mit 6 samples getestet (ich hatte einen run mit 200 samples bei der Evalauation fast durch doch ist sich dies zeitlich nicht mehr ausgegangen -> dort was das finetuned modell aber auch besser, wenn auch nur leicht). Wenn man die Antworten des finetuned Models sich genauer anschaut merkt man aber schnell, dass es übertrainiert wurde beim finetuning, denn viele Antworten folgen genau dem Muster des Trainingsdatensatz vom finetuning oder ergeben keinen Sinn mehr. Dies wird wohl daher stammen, dass das finetuning mit nur 500 samples durchgeführt wurde (es wurde early stopping angewendet beim finetuning). Beim finetuning sollten mehrere Datensätze kombineirt werden und mit einem viel größeren Datensatz sollte dann das finetuning durchgeführt werden. Die Evaluation lädt auf meinem Gerät extrem lange, da ich es nicht geschafft habe diese über CUDA laufen zu lassen (grundsätzlich wäre hier eine sample size von mind. 200 samples anzustreben).

Dieses Projekt zeigt nichtdestotrotz, dass durch ein relativ einfaches finetuning eine Annäherung der Antorten des Models zu den original responses in echten Konversationen.


---

**Autor:** [David Matischek]  
**Institution:** [Karl Franzens Universität Graz]  
**Datum:** August 2025  
**Kontakt:** [david.matischek@edu.uni-graz.at]
