import streamlit as st
import pandas as pd
import os
import time
import math
import plotly.express as px
import plotly.graph_objects as go
from scrape_fbref import scrape_league, LEAGUES

st.set_page_config(layout="wide", page_title="FBref Scraper Dashboard")

def load_data(league_key):
    """Loads appropriate CSV files for a given league."""
    data_dir = os.path.join("data", league_key)
    if not os.path.exists(data_dir):
        return None, None
    
    table_path = os.path.join(data_dir, f"{league_key}_table.csv")
    stats_path = os.path.join(data_dir, f"{league_key}_squad_stats.csv")
    
    df_table = pd.read_csv(table_path) if os.path.exists(table_path) else None
    df_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else None
    
    return df_table, df_stats

def clean_and_merge(df_table, df_stats):
    """Clean and merge league table with squad stats."""
    if df_table is None or df_stats is None:
        return None

    df_table = df_table.copy()
    df_stats = df_stats.copy()
    
    if "Squad" in df_table.columns:
        df_table["Squad"] = df_table["Squad"].astype(str).str.strip()
    
    if "Squad" in df_stats.columns:
        df_stats["Squad"] = df_stats["Squad"].astype(str).str.strip()

    df_merged = pd.merge(df_table, df_stats, on="Squad", how="inner", suffixes=("_table", "_stats"))
    
    df_merged = df_merged.drop_duplicates(subset=["Squad"])
    
    return df_merged

st.title("FBref Football Stats Scraper")

st.markdown("""
This application scrapes football data from FBref.com for the top 5 European leagues.
Now with **Advanced Analytics** and **Data Science Insights**!
""")

with st.sidebar:
    st.header("Settings")
    
    league_display_names = {k: k.replace("_", " ").title() for k in LEAGUES.keys()}
    selected_league_name = st.selectbox(
        "Choose League",
        options=list(league_display_names.values())
    )
    
    selected_league_key = [k for k, v in league_display_names.items() if v == selected_league_name][0]
    
    st.divider()

    if st.button("Run Scraping"):
        status_container = st.status("Starting scraping...", expanded=True)
        try:
            progress_bar = status_container.progress(0)
            total_leagues = len(LEAGUES)
            
            for i, (league_key, url) in enumerate(LEAGUES.items()):
                league_name_display = league_key.replace("_", " ").title()
                status_container.write(f"Scraping **{league_name_display}**...")
                scrape_league(league_key, url)
                
                progress_bar.progress((i + 1) / total_leagues)
                time.sleep(1) 
                
            status_container.update(label="Scraping completed successfully!", state="complete", expanded=False)
            st.success("Data updated!")
            
        except Exception as e:
            status_container.update(label="An error occurred during scraping.", state="error")
            st.error(f"Error: {str(e)}")

st.divider()

st.header(selected_league_name)

df_table, df_stats = load_data(selected_league_key)

if df_table is not None and df_stats is not None:
    
    df_merged = clean_and_merge(df_table, df_stats)

    if df_merged is not None:
        
        if "xG" not in df_merged.columns:
            if "xG_stats" in df_merged.columns:
                df_merged["xG"] = df_merged["xG_stats"]
            elif "xG_table" in df_merged.columns:
                df_merged["xG"] = df_merged["xG_table"]
        
        if "Gls" not in df_merged.columns:
            if "GF" in df_merged.columns:
                df_merged["Gls"] = df_merged["GF"]
            elif "Gls_stats" in df_merged.columns:
                 df_merged["Gls"] = df_merged["Gls_stats"]

        if "GA" not in df_merged.columns:
            if "GA_table" in df_merged.columns:
                df_merged["GA"] = df_merged["GA_table"]
        
        if "Poss" not in df_merged.columns:
            if "Poss_stats" in df_merged.columns:
                df_merged["Poss"] = df_merged["Poss_stats"]
        
        if "MP" not in df_merged.columns:
            if "MP_table" in df_merged.columns:
                df_merged["MP"] = df_merged["MP_table"]

        if "Pts" not in df_merged.columns:
            if "Pts_table" in df_merged.columns:
                df_merged["Pts"] = df_merged["Pts_table"]

        if "xGA" not in df_merged.columns:
            if "xGA_table" in df_merged.columns:
                df_merged["xGA"] = df_merged["xGA_table"]
        
        if "Attendance" not in df_merged.columns:
            if "Attendance_table" in df_merged.columns:
                df_merged["Attendance"] = df_merged["Attendance_table"]


        tab_dashboard, tab_charts, tab_comparison, tab_prediction, tab_report = st.tabs(["Dashboard", "Charts", "Comparison", "Prediction", "Report"])

        with tab_dashboard:
            st.markdown("### Top Generated Insights")
            cols_insight = st.columns(3)
            
            if "Gls" in df_merged.columns and "xG" in df_merged.columns:
                if "xG_Diff" not in df_merged.columns:
                     df_merged["xG_Diff"] = (df_merged["Gls"] - df_merged["xG"]).round(2)
                
                best_finisher = df_merged.loc[df_merged["xG_Diff"].idxmax()]
                with cols_insight[0]:
                    st.markdown("**Highest Offensive Efficiency**")
                    st.subheader(f"{best_finisher['Squad']}")
                    st.markdown(f"### :green[{best_finisher['xG_Diff']:.2f}]")
                    
                    st.markdown(f"""
                    The team scored **{best_finisher['Gls']}** goals against an expectation of **{best_finisher['xG']}** (xG).
                    """)

            if "Gls" in df_merged.columns and "xG" in df_merged.columns:
                worst_finisher = df_merged.loc[df_merged["xG_Diff"].idxmin()]
                with cols_insight[1]:
                    st.markdown("**Lowest Offensive Efficiency**")
                    st.subheader(f"{worst_finisher['Squad']}")
                    st.markdown(f"### :red[{worst_finisher['xG_Diff']:.2f}]")
                    
                    st.markdown(f"""
                    The team scored only **{worst_finisher['Gls']}** goals against an expectation of **{worst_finisher['xG']}** (xG).
                    """)
            
            if "GA" in df_merged.columns:
                    best_defense = df_merged.loc[df_merged["GA"].idxmin()]
                    with cols_insight[2]:
                        st.markdown("**Best Defense**")
                        st.subheader(f"{best_defense['Squad']}")
                        st.markdown(f"### :blue[{int(best_defense['GA'])}]")
                        
                        st.markdown(f"""
                        Goals conceded in the entire championship.
                        """)


        with tab_charts:
            with st.expander("View Raw Data (Table & Stats)", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("League Table")
                    st.dataframe(df_table, width="stretch")
                with col2:
                    st.subheader("Team Statistics")
                    st.dataframe(df_stats, width="stretch")

            st.markdown("### Interactive Insights")

            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.subheader("Goals vs Expected Goals (xG)")
                if "Gls" in df_merged.columns and "xG" in df_merged.columns:
                    df_merged["xG_Diff"] = (df_merged["Gls"] - df_merged["xG"]).round(2)
                    
                    df_merged["xG"] = pd.to_numeric(df_merged["xG"], errors='coerce').round(2)
                    df_merged["Gls"] = pd.to_numeric(df_merged["Gls"], errors='coerce')
                    df_merged = df_merged.dropna(subset=["xG", "Gls"])

                    if not df_merged.empty:
                        fig_xg = px.scatter(
                            df_merged, 
                            x="xG", 
                            y="Gls", 
                            color="xG_Diff",
                            hover_data=["Squad", "xG", "Gls", "xG_Diff"],
                            title="Offensive Efficiency (Color = Goals - xG)",
                            labels={"xG": "Expected Goals (xG)", "Gls": "Goals Scored", "xG_Diff": "Diff"},
                            color_continuous_scale="RdBu",
                            height=400
                        )
                        
                        min_val = min(df_merged["xG"].min(), df_merged["Gls"].min())
                        max_val = max(df_merged["xG"].max(), df_merged["Gls"].max())
                        fig_xg.add_shape(
                            type="line", line=dict(dash="dash", color="gray"),
                            x0=min_val, y0=min_val, x1=max_val, y1=max_val
                        )

                        st.plotly_chart(fig_xg, width="stretch")
                    else:
                        st.warning("Insufficient data to generate chart.")
                else:
                    st.warning("Columns 'xG' or 'Gls' not found.")

            with col_viz2:
                st.subheader("Possession vs Points")
                if "Poss" not in df_merged.columns and "Poss_stats" in df_merged.columns:
                    df_merged["Poss"] = df_merged["Poss_stats"]
                
                if "Pts" not in df_merged.columns and "Pts_table" in df_merged.columns:
                    df_merged["Pts"] = df_merged["Pts_table"]

                if "Poss" in df_merged.columns and "Pts" in df_merged.columns:
                    fig_poss = px.scatter(
                        df_merged,
                        x="Poss",
                        y="Pts",
                        color="Pts", 
                        hover_data=["Squad", "Poss", "Pts"],
                        title="Correlation: Possession vs Points",
                        labels={"Poss": "Possession (%)", "Pts": "Points"},
                        color_continuous_scale="Viridis",
                        height=400
                    )
                    st.plotly_chart(fig_poss, width="stretch")
                else:
                        st.warning("Columns 'Poss' or 'Pts' not found.")

            st.subheader("Goal Difference (GF - GA)")
            if "GF" in df_merged.columns and "GA" in df_merged.columns:
                df_merged["GoalDiff"] = df_merged["GF"] - df_merged["GA"]
                df_merged_sorted = df_merged.sort_values(by="GoalDiff", ascending=False)
                
                fig_gd = px.bar(
                    df_merged_sorted,
                    x="Squad",
                    y="GoalDiff",
                    color="GoalDiff",
                    color_continuous_scale="RdBu",
                    title="Goal Difference by Team"
                )
                st.plotly_chart(fig_gd, width="stretch")

            st.divider()
            col_def, col_att = st.columns(2)
            
            with col_def:
                st.subheader("Defensive Performance: xGA vs GA")
                if "xGA" in df_merged.columns and "GA" in df_merged.columns:
                     df_merged["xGA"] = pd.to_numeric(df_merged["xGA"], errors='coerce')
                     df_merged["GA"] = pd.to_numeric(df_merged["GA"], errors='coerce')
                     df_merged["Def_Diff"] = (df_merged["xGA"] - df_merged["GA"]).round(2)
                     
                     fig_def = px.scatter(
                        df_merged,
                        x="xGA",
                        y="GA",
                        color="Def_Diff",
                        hover_data=["Squad", "xGA", "GA", "Def_Diff"],
                        title="Expected Goals Against (xGA) vs Goals Conceded (GA)",
                        labels={"xGA": "Expected Goals Against (xGA)", "GA": "Goals Conceded (GA)", "Def_Diff": "Def. Diff"},
                        color_continuous_scale="RdBu_r", 
                        color_continuous_midpoint=0
                     )
                     
                     min_def = min(df_merged["xGA"].min(), df_merged["GA"].min())
                     max_def = max(df_merged["xGA"].max(), df_merged["GA"].max())
                     fig_def.add_shape(
                        type="line", line=dict(dash="dash", color="white"),
                        x0=min_def, y0=min_def, x1=max_def, y1=max_def
                     )
                     
                     st.plotly_chart(fig_def, width="stretch")
                     st.info("Above dotted line: Conceded more than expected. Below: Conceded less.")
                else:
                    st.warning("xGA or GA data not found.")

            with col_att:
                st.subheader("Average Attendance")
                if "Attendance" in df_merged.columns:
                     if df_merged["Attendance"].dtype == object:
                        df_merged["Attendance"] = df_merged["Attendance"].astype(str).str.replace(",", "").astype(float)
                     
                     df_att_sorted = df_merged.sort_values(by="Attendance", ascending=True) 
                     
                     fig_att = px.bar(
                        df_att_sorted,
                        x="Attendance",
                        y="Squad",
                        orientation='h',
                        title="Average Attendance by Stadium",
                        labels={"Attendance": "Avg Attendance", "Squad": "Team"},
                        color="Attendance",
                        color_continuous_scale="Blues",
                        height=600
                     )
                     fig_att.update_layout(yaxis={'categoryorder':'total ascending'})
                     
                     st.plotly_chart(fig_att, width="stretch")
                else:
                    st.warning("Attendance data not found.")

        with tab_comparison:
            st.subheader("Team Comparison (Radar)")
            
            col_sel1, col_sel2 = st.columns(2)
            teams_list = df_merged["Squad"].sort_values().unique()
            
            with col_sel1:
                team_a = st.selectbox("Select Team A", options=teams_list, index=0)
            with col_sel2:
                default_idx = 1 if len(teams_list) > 1 else 0
                team_b = st.selectbox("Select Team B", options=teams_list, index=default_idx)

            metrics_config = {
                "Gls": "Goals Scored",
                "xG": "xG (Creativity)",
                "Poss": "Possession",
                "Ast": "Assists",
                "GA": "Defense (Inv)" 
            }
            
            available_metrics = [m for m in metrics_config.keys() if m in df_merged.columns]
            
            if len(available_metrics) >= 3:
                df_norm = df_merged.copy()
                
                if "Poss" in df_norm.columns:
                    if df_norm["Poss"].dtype == object:
                         df_norm["Poss"] = df_norm["Poss"].astype(str).str.replace("%","").astype(float)
                
                for col in available_metrics:
                    df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
                    
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    
                    if max_val - min_val != 0:
                        if col == "GA":
                            df_norm[f"{col}_norm"] = (max_val - df_norm[col]) / (max_val - min_val)
                        else:
                            df_norm[f"{col}_norm"] = (df_norm[col] - min_val) / (max_val - min_val)
                    else:
                         df_norm[f"{col}_norm"] = 0.5 
                
                data_a = df_norm[df_norm["Squad"] == team_a]
                data_b = df_norm[df_norm["Squad"] == team_b]
                
                if not data_a.empty and not data_b.empty:
                    categories = [metrics_config[m] for m in available_metrics]
                    
                    values_a = [data_a[f"{m}_norm"].values[0] for m in available_metrics]
                    values_b = [data_b[f"{m}_norm"].values[0] for m in available_metrics]
                    
                    raw_a = [data_a[m].values[0] for m in available_metrics]
                    raw_b = [data_b[m].values[0] for m in available_metrics]

                    fig_radar = go.Figure()

                    fig_radar.add_trace(go.Scatterpolar(
                        r=values_a,
                        theta=categories,
                        fill='toself',
                        name=team_a,
                        text=[f"{v:.2f}" for v in raw_a],
                        hoverinfo="text+name+theta"
                    ))
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values_b,
                        theta=categories,
                        fill='toself',
                        name=team_b,
                        text=[f"{v:.2f}" for v in raw_b],
                        hoverinfo="text+name+theta"
                    ))

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        title="Relative Comparison (0-1 Scale)"
                    )

                    st.plotly_chart(fig_radar, width="stretch")
                    st.info("'Defense (Inv)' means higher value = fewer goals conceded (better defense).")

                else:
                    st.error("Error loading data for selected teams.")
            else:
                 st.warning("Insufficient metrics (Gls, xG, GA...) to generate radar.")

        with tab_prediction:
            st.subheader("Match Simulator (Poisson Model)")
            st.markdown("This model uses a custom implementation compatible with **Scikit-Learn**.")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pred_team_a = st.selectbox("Home Team", options=teams_list, index=0, key="pred_a")
                
            with col_p2:
                def_idx_b = 1 if len(teams_list) > 1 else 0
                pred_team_b = st.selectbox("Away Team", options=teams_list, index=def_idx_b, key="pred_b")

            if st.button("Simulate Match"):
                try:
                    from prediction_model import PoissonMatchPredictor
                    
                    predictor = PoissonMatchPredictor()
                    predictor.fit(df_merged)
                    
                    home_adv_factor = 1.15 
                    result = predictor.predict_match(pred_team_a, pred_team_b, home_advantage=home_adv_factor)
                    
                    if result:
                        lambda_a = result['lambda_home']
                        lambda_b = result['lambda_away']
                        win_a = result['home_win']
                        draw = result['draw']
                        win_b = result['away_win']

                        st.divider()
                        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                        
                        with col_res1:
                            st.markdown(f"""
                                <div style="text-align: center;">
                                    <h3 style="margin-bottom: 0px;">{pred_team_a}</h3>
                                    <div style="display: flex; justify-content: center; align-items: baseline; gap: 8px;">
                                        <h1 style="margin-top: 0px; margin-bottom: 0px;">{lambda_a:.2f}</h1>
                                        <span style="color: gray; font-size: 0.8em;">Expected xG</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col_res2:
                            st.markdown("<h4 style='text-align: center'>Probabilities</h4>", unsafe_allow_html=True)
                            
                            st.write(f"**Win {pred_team_a}:** {win_a*100:.1f}%")
                            st.progress(win_a)
                            
                            st.write(f"**Draw:** {draw*100:.1f}%")
                            st.progress(draw)
                            
                            st.write(f"**Win {pred_team_b}:** {win_b*100:.1f}%")
                            st.progress(win_b)

                        with col_res3:
                            st.markdown(f"""
                                <div style="text-align: center;">
                                    <h3 style="margin-bottom: 0px;">{pred_team_b}</h3>
                                    <div style="display: flex; justify-content: center; align-items: baseline; gap: 8px;">
                                        <h1 style="margin-top: 0px; margin-bottom: 0px;">{lambda_b:.2f}</h1>
                                        <span style="color: gray; font-size: 0.8em;">Expected xG</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.info("Model trained in `prediction_model.py`.")
                    else:
                        st.error("Prediction error: Teams not found.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with tab_report:
            st.markdown("## Technical Project Report")
            st.markdown("This section details the technical architecture, stack, and methodologies used in this project. Designed to demonstrate end-to-end Data Science competencies.")
            
            st.divider()

            col_tech1, col_tech2 = st.columns(2)
            
            with col_tech1:
                st.subheader("Tech Stack")
                st.markdown("""
                - **Language:** Python 3.10+
                - **Data Acquisition:** `curl_cffi`, `BeautifulSoup4`
                - **Data Processing:** `Pandas`, `NumPy`
                - **Machine Learning:** `Scikit-Learn` (Base Estimator), `SciPy` (Stats)
                - **Visualization:** `Plotly Express`, `Plotly Graph Objects`
                - **Web Framework:** `Streamlit`
                """)
                
            with col_tech2:
                st.subheader("Architecture Pipeline")
                st.code("""
[FBref.com] 
    ⬇ (HTTP/TLS 1.3 | curl_cffi)
[Scraper Engine]
    ⬇ (HTML Parsing & Comment Extraction)
[Raw CSV Storage]
    ⬇ (Data Cleaning & Merging)
[Pandas DataFrame] ➡ [Poisson Model]
    ⬇
[Interactive Dashboard]
                """)

            st.divider()

            st.subheader("Data Science Methodologies")

            with st.expander("1. Robust Web Scraping Strategy", expanded=True):
                st.markdown("""
                **Challenge:** Modern websites often use TLS fingerprinting to block automated scrapers. Additionally, FBref obfuscates data tables by hiding them inside HTML comments `<!-- -->` to prevent basic scraping.
                
                **Solution:**
                - **Bypassing Protections:** Implemented `curl_cffi` to impersonate a real Chrome browser fingerprint (TLS Client Hello), drastically reducing 403 Forbidden errors.
                - **Parsing Hidden Data:** Created a custom parsing logic that specifically searches for `bs4.Comment` objects containing string `"<table"`. These comments are then parsed as separate IO streams using `pd.read_html`.
                
                *See `scrape_fbref.py` for implementation details.*
                """)

            with st.expander("2. Predictive Modeling (Poisson Distribution)", expanded=True):
                st.markdown(r"""
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
                """)

            st.info("**Note for Recruiters:** This project demonstrates ability in Data Engineering (ETL), Mathematical Modeling, and Full-Stack Data Application development.")

else:
        st.info(f"Data not found for {selected_league_name}. Please run scraping first.")
