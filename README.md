# Record Label Event Extraction and Visualization

This repository contains the full workflow of my master's research project, which applies **Natural Language Processing (NLP)** techniques to crowdsourced music industry data from **Discogs**.  
The project builds an automated pipeline for extracting structured historical events from free-text label descriptions and provides an **interactive dashboard** for analysis.

## Project Objectives
- Extract record label historical events (Founded, Merged, Discontinued, Renamed) from large-scale Discogs data.
- Construct structured company timelines with confidence scores and temporal normalization.
- Build an interactive visualization system for researchers and potential industry investors.

## Components
- **NLP Pipeline (spaCy 3.x + custom rules)**  
  - Data cleaning and normalization  
  - Entity recognition (label names, time expressions)  
  - Relation extraction (dependency parsing + pattern matching)  
  - Timeline construction and conflict resolution  

- **Dashboard (Streamlit + Plotly)**  
  - Upload CSV/Parquet datasets  
  - KPI cards (Total companies, Active companies, Average lifespan, Total events)  
  - Event trends line/area charts  
  - Heatmap of industry evolution (year, 5-year, 10-year groupings)  
  - Top companies ranking table with download option  
  - Data explorer with search and export  

## Dataset
- Source: **Discogs monthly XML data dump (May 2025)**  
- Size: 2.43 million record labels worldwide  
- Main analysis field: `profile` (free-text historical descriptions)

## ⚙Installation
```bash
git clone https://github.com/your-username/record-label-event-extraction.git
cd record-label-event-extraction
pip install -r requirements.txt
