from curl_cffi import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import sys
import io
import time
import os

LEAGUES = {
    "premier_league": "https://fbref.com/en/comps/9/Premier-League-Stats",
    "la_liga": "https://fbref.com/en/comps/12/La-Liga-Stats",
    "ligue_1": "https://fbref.com/en/comps/13/Ligue-1-Stats",
    "bundesliga": "https://fbref.com/en/comps/20/Bundesliga-Stats",
    "serie_a": "https://fbref.com/en/comps/11/Serie-A-Stats"
}

def scrape_league(league_name, url):
    print(f"Fetching data for {league_name} from {url}...")
    try:
        response = requests.get(url, impersonate="chrome110", timeout=30)
        
        if response.status_code != 200:
            print(f"Failed with status code: {response.status_code}")
            return
            
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    
    dfs_all = []
    
    try:
        tables = pd.read_html(io.StringIO(str(soup)))
        dfs_all.extend(tables)
    except ValueError:
        pass 

    for c in comments:
        if "<table" in c:
            try:
                tables = pd.read_html(io.StringIO(c))
                dfs_all.extend(tables)
            except ValueError:
                continue

    print(f"Found {len(dfs_all)} tables in total for {league_name}.")
    
    output_dir = os.path.join("data", league_name)
    os.makedirs(output_dir, exist_ok=True)

    league_table = None
    for df in dfs_all:
        if set(["Squad", "MP", "W", "D", "L", "Pts"]).issubset(df.columns) or \
           (isinstance(df.columns, pd.MultiIndex) and "Squad" in df.columns.get_level_values(0)):
             
             if isinstance(df.columns, pd.MultiIndex):
                 cols = df.columns.get_level_values(-1)
             else:
                 cols = df.columns
             
             if "Squad" in cols and "Pts" in cols and "MP" in cols:
                 if "Rk" in cols or "Rank" in cols:
                     league_table = df
                     break
    

    if league_table is not None:
        print(f"Found Overall League Table for {league_name}.")
        if isinstance(league_table.columns, pd.MultiIndex):
            league_table.columns = league_table.columns.get_level_values(-1)
        filename = os.path.join(output_dir, f"{league_name}_table.csv")
        league_table.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    else:
        print(f"Could not find Overall League Table for {league_name}.")

    squad_stats = None
    for df in dfs_all:
         if isinstance(df.columns, pd.MultiIndex):
             cols = df.columns.get_level_values(-1)
         else:
             cols = df.columns
             
         target_cols = ["Gls", "Ast", "npxG"]
         if set(target_cols).issubset(cols) and "Squad" in cols:
             squad_stats = df
             break
    
    if squad_stats is not None:
        print(f"Found Squad Standard Stats for {league_name}.")
        if isinstance(squad_stats.columns, pd.MultiIndex):
            squad_stats.columns = squad_stats.columns.get_level_values(-1)
        filename = os.path.join(output_dir, f"{league_name}_squad_stats.csv")
        squad_stats.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    else:
        print(f"Could not find Squad Standard Stats for {league_name}.")

def main():
    for league_name, url in LEAGUES.items():
        scrape_league(league_name, url)
        print("-" * 30)
        time.sleep(5) 

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
