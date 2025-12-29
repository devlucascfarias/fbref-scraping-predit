# Technical Project Report

This section details the technical architecture, stack, and methodologies used in this project. Designed to demonstrate end-to-end Data Science competencies.

## Tech Stack
- **Language:** Python 3.10+
- **Data Acquisition:** `curl_cffi`, `BeautifulSoup4`
- **Data Processing:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-Learn` (Base Estimator), `SciPy` (Stats)
- **Visualization:** `Plotly Express`, `Plotly Graph Objects`
- **Web Framework:** `Streamlit`

## Architecture Pipeline
```
[FBref.com] 
    ⬇ (HTTP/TLS 1.3 | curl_cffi)
[Scraper Engine]
    ⬇ (HTML Parsing & Comment Extraction)
[Raw CSV Storage]
    ⬇ (Data Cleaning & Merging)
[Pandas DataFrame] ➡ [Poisson Model]
    ⬇
[Interactive Dashboard]
```

## Data Science Methodologies

### 1. Robust Web Scraping Strategy
**Challenge:** Modern websites often use TLS fingerprinting to block automated scrapers. Additionally, FBref obfuscates data tables by hiding them inside HTML comments `<!-- -->` to prevent basic scraping.

**Solution:**
- **Bypassing Protections:** Implemented `curl_cffi` to impersonate a real Chrome browser fingerprint (TLS Client Hello), drastically reducing 403 Forbidden errors.
- **Parsing Hidden Data:** Created a custom parsing logic that specifically searches for `bs4.Comment` objects containing string `"<table"`. These comments are then parsed as separate IO streams using `pd.read_html`.

*See `scrape_fbref.py` for implementation details.*

### 2. Predictive Modeling (Poisson Distribution)
**Objective:** Predict match outcomes based on historical team performance.

**Theory:** Football goals are rare, independent events that strongly follow a **Poisson Distribution**.

**Algorithm:**
1. **Metric Calculation:** For every team, we calculate an **Attack Strength** (Goals Scored / League Avg) and **Defense Strength** (Goals Conceded / League Avg).
2. **Expected Goals ($\lambda$):** For a match between Team A (Home) and Team B (Away):
   $$ \lambda_{Home} = \text{Att}_A \times \text{Def}_B \times \text{LeagueAvg} \times \text{HomeAdvantage} $$
   $$ \lambda_{Away} = \text{Att}_B \times \text{Def}_A \times \text{LeagueAvg} $$
3. **Probability Matrix:** We simulate the probability of every possible scoreline (0-0, 1-0, ... 5-5) using the Probability Mass Function (PMF):
   $$ P(k) = \frac{\lambda^k e^{-\lambda}}{k!} $$
4. **Outcome Aggregation:** Summing probabilities where $Goals_A > Goals_B$ gives the Home Win % (and vice versa).

*See `prediction_model.py` for the custom Scikit-Learn Estimator.*

> **Note for Recruiters:** This project demonstrates ability in Data Engineering (ETL), Mathematical Modeling, and Full-Stack Data Application development.
