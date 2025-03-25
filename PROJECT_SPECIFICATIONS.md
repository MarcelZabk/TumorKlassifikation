# Projekt Spezifikation: Tumor Klassifikation mit KI

## Einleitung
Dieses Projekt entstand aus dem Interesse, moderne KI-Technologien in der medizinischen Diagnostik zu erforschen. Das Ziel war es, ein Machine-Learning-Modell zu entwickeln, das auf Basis verschiedener Merkmale von Brustkrebszellen zwischen gutartigen und bösartigen Tumoren unterscheiden kann. Obwohl es sich um ein Demonstrationsprojekt handelt, zeigt es das Potenzial von KI-Systemen in der medizinischen Diagnostik.

Die Idee war, ein System zu entwickeln, das die Analyse von Tumordaten unterstützt und möglicherweise einen Beitrag zur Früherkennung leisten kann. Wichtig zu betonen ist, dass dieses System keinesfalls eine ärztliche Diagnose ersetzt, sondern als unterstützendes Tool gedacht ist.

## Zielsetzung
Das Hauptziel des Projekts war die Entwicklung eines präzisen und effizienten Machine-Learning-Modells, das basierend auf 30 verschiedenen Merkmalen von Brustkrebszellen zwischen gutartigen und bösartigen Tumoren unterscheidet. Ein besonderer Fokus lag auf der Genauigkeit der Vorhersagen und der Interpretierbarkeit der Ergebnisse.

## Datensatz
Der Datensatz stammt aus der scikit-learn Bibliothek (Breast Cancer Wisconsin Dataset) und enthält folgende Informationen:
- 569 Proben
- 30 verschiedene Merkmale pro Probe
- Binäre Klassifikation (0 = gutartig, 1 = bösartig)

Die 30 Merkmale umfassen verschiedene Messungen der Zellkerne, wie:
- Radius (Mittelwert, Standardabweichung, "worst case")
- Texture (Mittelwert, Standardabweichung, "worst case")
- Perimeter (Mittelwert, Standardabweichung, "worst case")
- Area (Mittelwert, Standardabweichung, "worst case")
- Smoothness (Mittelwert, Standardabweichung, "worst case")
- Compactness (Mittelwert, Standardabweichung, "worst case")
- Concavity (Mittelwert, Standardabweichung, "worst case")
- Concave points (Mittelwert, Standardabweichung, "worst case")
- Symmetry (Mittelwert, Standardabweichung, "worst case")
- Fractal dimension (Mittelwert, Standardabweichung, "worst case")

## Vorverarbeitung
Die Daten wurden wie folgt aufbereitet:
- Standardisierung der Merkmale (StandardScaler)
- Aufteilung der Daten:
  - Trainingsdaten: 80% des Datensatzes
  - Testdaten: 20% des Datensatzes
- Konvertierung in PyTorch-Tensoren für das Training

## Modellarchitektur
Das Modell wurde als neuronales Netzwerk mit folgenden Schichten implementiert:
- Eingabeschicht: 30 Neuronen (für die 30 Merkmale)
- Versteckte Schicht: 64 Neuronen mit ReLU-Aktivierungsfunktion
- Versteckte Schicht: 32 Neuronen mit ReLU-Aktivierungsfunktion
- Ausgabeschicht: 1 Neuron mit Sigmoid-Aktivierungsfunktion

## Hyperparameter
- Optimierer: Adam
- Lernrate: 0.001
- Verlustfunktion: Binary Cross Entropy (BCELoss)
- Epochen: 50
- Batch-Größe: Vollständiger Datensatz (Full Batch Training)

## Modellleistung
Das Modell erreicht typischerweise:
- Testgenauigkeit: > 95%
- Präzise Unterscheidung zwischen gutartigen und bösartigen Tumoren
- Detaillierte Ausgabe von:
  - Vorhergesagter Klasse
  - Konfidenz der Vorhersage
  - Wahrscheinlichkeitsverteilung
  - Falsch-Positiv und Falsch-Negativ Raten

## Implementierung
Das Projekt wurde in Python implementiert und verwendet:
- PyTorch für das neuronale Netzwerk
- scikit-learn für Datenvorverarbeitung und Metriken
- NumPy für numerische Operationen
- joblib für das Speichern und Laden des Modells

## Projektstruktur
```
TumorKlassifikation/
├── src/
│   ├── models/
│   │   └── tumor_classifier.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── tumor_classifier.pth
│   └── scaler.pkl
├── requirements.txt
└── README.md
``` 