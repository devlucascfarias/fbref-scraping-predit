import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.stats import poisson

class PoissonMatchPredictor(BaseEstimator):
    """
    A Scikit-Learn compatible estimator for predicting football match outcomes
    using a Poisson distribution based on team attack and defense strengths.
    """
    def __init__(self):
        self.league_avg_goals_ = None
        self.team_stats_ = {}
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the model using league summary data.
        
        Parameters:
        X (pd.DataFrame): DataFrame containing 'Squad', 'GF', 'GA', 'MP' (or equivalents).
        y: Ignored.
        """
        df = X.copy()
        
        col_mp = "MP_table" if "MP_table" in df.columns else ("MP" if "MP" in df.columns else None)
        col_gf = "GF" if "GF" in df.columns else ("GF_table" if "GF_table" in df.columns else ("Gls" if "Gls" in df.columns else None))
        col_ga = "GA" if "GA" in df.columns else ("GA_table" if "GA_table" in df.columns else None)
        
        if not (col_mp and col_gf and col_ga):
            raise ValueError(f"Required columns (MP, GF, GA) not found. Columns present: {df.columns}")

        total_goals = df[col_gf].sum()
        total_matches = df[col_mp].sum()
        
        if total_matches == 0:
            raise ValueError("Total matches is zero, cannot calculate averages.")
            
        self.league_avg_goals_ = total_goals / total_matches
        
        for _, row in df.iterrows():
            squad = row["Squad"]
            mp = row[col_mp]
            gf = row[col_gf]
            ga = row[col_ga]
            
            if mp > 0:
                att_strength = (gf / mp) / self.league_avg_goals_
                def_strength = (ga / mp) / self.league_avg_goals_
            else:
                att_strength = 1.0 
                def_strength = 1.0
                
            self.team_stats_[squad] = {
                "att": att_strength,
                "def": def_strength
            }
            
        self.is_fitted_ = True
        return self

    def predict_match(self, home_team, away_team, home_advantage=1.0, max_goals=6):
        """
        Predict probabilities for a specific match.
        
        Parameters:
        home_advantage (float): Multiplier for home team attack strength (e.g. 1.1 = 10% boost).
        
        Returns:
        dict: {'home_win': float, 'draw': float, 'away_win': float, 'lambda_home': float, 'lambda_away': float}
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
            
        if home_team not in self.team_stats_ or away_team not in self.team_stats_:
            return None 
            
        home_stats = self.team_stats_[home_team]
        away_stats = self.team_stats_[away_team]
        
        lambda_home = home_stats["att"] * away_stats["def"] * self.league_avg_goals_ * home_advantage
        lambda_away = away_stats["att"] * home_stats["def"] * self.league_avg_goals_
        
        probs_home = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
        probs_away = [poisson.pmf(i, lambda_away) for i in range(max_goals)]
        
        prob_home_win = 0
        prob_draw = 0
        prob_away_win = 0
        
        for g_h in range(max_goals):
            for g_a in range(max_goals):
                p = probs_home[g_h] * probs_away[g_a]
                
                if g_h > g_a:
                    prob_home_win += p
                elif g_h == g_a:
                    prob_draw += p
                else:
                    prob_away_win += p
                    
        return {
            "home_win": prob_home_win,
            "draw": prob_draw,
            "away_win": prob_away_win,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away
        }
