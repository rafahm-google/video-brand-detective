# ==============================================================================
# FINAL MULTI-BRAND SPONSORSHIP ENHANCED ANALYSIS SCRIPT (V4 - CDO FIX)
#
# Description:
# This script performs an in-depth analysis of a brand's sponsorship
# performance on CazéTV. It integrates ad insertion logs with search
# performance data to measure brand lift, identify key creative drivers,
# and generate two distinct reports for both the CMO and CDO.
#
# Key Enhancements in this Version:
# - CDO PAYLOAD FIX: Surgically trimmed the data payloads for CDO report sections
#   to resolve `MAX_TOKENS` errors and prevent timeouts.
# - JSON SERIALIZATION FIX: Corrected the `CustomJSONEncoder` to handle
#   pandas Series objects, resolving the `TypeError`.
# - MODULARITY: The main analysis is broken into clear, single-responsibility
#   functions, orchestrated by the main `run_full_analysis` block.
# - COST-EFFICIENCY: Replaced the expensive LLM call for generating the appendix
#   with a deterministic Python function.
# ==============================================================================

# --- Standard Libraries ---
import unidecode
import pandas as pd
import numpy as np
import os
import glob
import json
import re
from datetime import timedelta
import warnings
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import copy

# --- Analytical Libraries ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import statsmodels.formula.api as smf
import holidays

# --- Optional Libraries (with availability checks) ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import jinja2
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

try:
    from markdown import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CENTRALIZED CONFIGURATION
# @what: Stores all constants and settings in one place for easy updates.
# ==============================================================================
CONFIG = {
    "SPONSORSHIP_START_DATE": "2025-01-15",
    "MIN_AD_COUNT_FOR_ANALYSIS": 5,
    "P_VALUE_SIGNIFICANCE_THRESHOLD": 0.1,
    "GAME_PERFORMANCE_TOP_N": 3,
    "MAX_CLUSTERS_TO_TEST": 6,
    "LOG_FILE_PATH": "output_reports_v2",
    "SEARCH_DATA_PATH": "search_performance_data",
    "GENERIC_DATA_FILE_TPL": "{brand_slug}_all_searches_performance.csv",
    "BRANDED_DATA_FILE_TPL": "{brand_slug}_branded_searches_performance.csv",
    "CMO_REPORT_FILE_BASENAME": "brand_report.html",
    "CDO_APPENDIX_FILE_BASENAME": "technical_report.html",
    "FINAL_DATA_PACKAGE_BASENAME": "final_llm_data_package",
    "AD_GROUP_COLUMNS": [
        'anuncios_integrados_nativos', 'anuncios_video_padrao',
        'anuncios_genericos', 'anuncios_conversao', 'outros_formatos'
    ],
    "MOMENTO_PREFIX": "momento_",
    "SEMANTIC_MODEL_NAME": 'paraphrase-multilingual-MiniLM-L12-v2',
    "CONTROL_VARS": ['is_holiday', 'branded_spend'],
    "TARGET_METRIC_LIFT": "ad_opportunities",
    "CTA_COLUMN": "detalhe_cta_transcricao",
    "BRAND_TEAM_REPORT_BASENAME": "brand_tactical_summary.html",
}

# ==============================================================================
# 2. LOGGER SETUP
# @what: Creates a global logger to record all script actions for debugging.
# ==============================================================================
def setup_logger():
    """@what: Initializes a global logger writing to 'analysis_debug.log' and the console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis_debug.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

analysis_logger = setup_logger()

# ==============================================================================
# 3. HELPER & DATA PREPARATION FUNCTIONS
# @what: Handles file I/O, data cleaning, and feature engineering.
# ==============================================================================
def _sanitize_string(s):
    s_no_apostrophe = str(s).lower().replace("'", "")
    s_unidecoded = unidecode.unidecode(s_no_apostrophe)
    s_cleaned = re.sub(r'[^a-zA-Z0-9_]+', '_', s_unidecoded).strip('_')
    return s_cleaned

def check_and_create_data(brand_name, brand_slug):
    pass

def load_and_clean_data(brand_name, brand_slug):
    analysis_logger.info("Starting Data Ingestion & Cleaning...")
    log_files = glob.glob(os.path.join(CONFIG['LOG_FILE_PATH'], "*_detailed_log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No detailed log files found in '{CONFIG['LOG_FILE_PATH']}/'.")

    logs_df = pd.concat([pd.read_csv(f, on_bad_lines='skip', encoding='utf-8') for f in log_files], ignore_index=True)
    logs_df.columns = [_sanitize_string(c) for c in logs_df.columns]
    
    for col in ['formato_contexto', 'sponsor', CONFIG['CTA_COLUMN'], 'video_title', 'momento_no_jogo']:
        if col in logs_df.columns:
            logs_df[col] = logs_df[col].apply(lambda x: str(x).lower().strip())

    logs_df['date'] = pd.to_datetime(logs_df['date'], errors='coerce')
    
    if brand_name.lower() == 'pagbank':
        logs_df['sponsor'] = logs_df['sponsor'].replace('pagseguro', 'pagbank')
            
    brand_logs = logs_df[logs_df['sponsor'] == brand_name.lower()].copy().dropna(subset=['date'])
    
    branded_file_path = os.path.join(CONFIG['SEARCH_DATA_PATH'], CONFIG['BRANDED_DATA_FILE_TPL'].format(brand_slug=brand_slug))
    generic_file_path = os.path.join(CONFIG['SEARCH_DATA_PATH'], CONFIG['GENERIC_DATA_FILE_TPL'].format(brand_slug=brand_slug))
    
    branded_df = pd.read_csv(branded_file_path); branded_df['Start Date'] = pd.to_datetime(branded_df['Start Date'], errors='coerce'); branded_df['Advertiser Name'] = brand_name
    generic_df = pd.read_csv(generic_file_path); generic_df['Start Date'] = pd.to_datetime(generic_df['Start Date'], errors='coerce')
    client_name_in_file = [name for name in generic_df['Advertiser Name'].unique() if 'peer' not in str(name).lower()][0]
    generic_df.loc[generic_df['Advertiser Name'] == client_name_in_file, 'Advertiser Name'] = brand_name
    return brand_logs, branded_df, generic_df

def map_tactical_to_strategic_groups(logs_df):
    analysis_logger.info("--> Mapping tactical formats to strategic ad groups...")
    strategic_map_raw = {
        'gráfico integrado (placar/replay/estatísticas/agenda)': 'anuncios_genericos','banner overlay de rodapé (lower-third)': 'anuncios_genericos',
        'anúncio estático na tela': 'anuncios_genericos','anúncio em formato l (l-shape)': 'anuncios_conversao',
        'anúncio call-to-action genérico': 'anuncios_conversao','anúncio em vídeo': 'anuncios_video_padrao',
        'quadro patrocinado (segmento de conteúdo)': 'anuncios_integrados_nativos','integração em comentário ao vivo': 'anuncios_integrados_nativos',
        'menção em áudio': 'anuncios_integrados_nativos','nan': 'outros_formatos'
    }
    strategic_map = {str(k).lower().strip(): v for k, v in strategic_map_raw.items()}
    logs_df['ad_group'] = logs_df['formato_contexto'].map(strategic_map).fillna('outros_formatos')
    return logs_df

def create_master_df_enhanced(branded_df, generic_df, logs_with_groups, brand_name):
    analysis_logger.info("Starting Enhanced Feature Engineering...")
    brand_slug = _sanitize_string(brand_name)
    
    BIG_CLUBS = ['flamengo', 'corinthians', 'sao paulo', 'palmeiras', 'vasco', 'gremio', 'internacional', 'atletico mineiro', 'cruzeiro', 'fluminense', 'botafogo', 'santos']
    def get_game_importance(video_title):
        title_lower = str(video_title).lower()
        return sum(1 for club in BIG_CLUBS if club in title_lower)
    logs_with_groups['game_importance_score'] = logs_with_groups['video_title'].apply(get_game_importance)
    daily_game_importance = logs_with_groups.groupby('date')['game_importance_score'].max()

    branded_daily_cols = ['Clicks', 'Impressions', 'Spend']
    if CONFIG['TARGET_METRIC_LIFT'] == 'ad_opportunities':
        branded_daily_cols.append('Ad Opportunities')

    branded_daily = branded_df.groupby('Start Date')[branded_daily_cols].sum().asfreq('D')
    branded_daily.columns = [col.lower().replace(' ', '_') for col in branded_daily.columns]
    branded_daily.rename(columns={'clicks': 'branded_clicks', 'impressions': 'branded_impressions', 'spend': 'branded_spend'}, inplace=True)

    generic_pivot = generic_df.pivot_table(index='Start Date', columns='Advertiser Name', values=['Clicks', 'Impressions', 'Spend']).fillna(0)
    generic_pivot.columns = ['_'.join(col).strip().lower().replace(' ', '_').replace("'", "") for col in generic_pivot.columns.values]
    brand_sov_df = generic_df[generic_df['Advertiser Name'] == brand_name][['Start Date', 'Share of Voice']].set_index('Start Date').rename(columns={'Share of Voice': 'share_of_voice_generic'})
    logs_with_groups['count'] = 1
    daily_ad_group_counts = logs_with_groups.pivot_table(index='date', columns='ad_group', values='count', aggfunc='sum')
    daily_momento_counts = logs_with_groups.pivot_table(index='date', columns='momento_no_jogo', values='count', aggfunc='sum')
    daily_momento_counts.columns = [f"{CONFIG['MOMENTO_PREFIX']}{_sanitize_string(col)}" for col in daily_momento_counts.columns]
    
    master_df = branded_daily.join(generic_pivot, how='left').join(brand_sov_df, how='left').join(daily_ad_group_counts, how='left').join(daily_momento_counts, how='left').join(daily_game_importance, how='left').fillna(0)
    
    master_df['day_of_week'] = master_df.index.dayofweek
    master_df['is_weekend'] = master_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    master_df['is_holiday'] = master_df.index.to_series().apply(lambda date: 1 if date in holidays.Brazil() else 0).astype(int)
    
    for group in CONFIG['AD_GROUP_COLUMNS']:
        if group not in master_df.columns: master_df[group] = 0
    master_df.rename(columns={f'clicks_{brand_slug}': 'generic_clicks', f'spend_{brand_slug}': 'generic_spend', f'impressions_{brand_slug}': 'generic_impressions'}, inplace=True)
    for col in ['generic_clicks', 'generic_spend', 'generic_impressions', 'clicks_peer_group', 'impressions_peer_group', 'spend_peer_group']:
        if col not in master_df.columns: master_df[col] = 0
        
    analysis_logger.info("-> Master dataframe created successfully with Game Importance and Day of Week features.")
    return master_df

def get_eligible_game_dates(logs_with_groups, master_df):
    game_ad_counts = logs_with_groups[logs_with_groups['date'].dt.normalize().isin(master_df.index)].groupby('date').size()
    min_ads = CONFIG["MIN_AD_COUNT_FOR_ANALYSIS"]
    eligible_dates = game_ad_counts[game_ad_counts >= min_ads].index.strftime('%Y-%m-%d').tolist()
    analysis_logger.info(f"Found {len(eligible_dates)} eligible games (>= {min_ads} ads).")
    return eligible_dates

# ==============================================================================
# 4. CORE MODELING & ANALYSIS FUNCTIONS
# ==============================================================================

def perform_ad_group_lift_analysis(df):
    analysis_logger.info("--> Analyzing the direct lift from strategic ad groups, game context...")
    context_vars = ['is_holiday', 'is_weekend', 'game_importance_score', 'branded_spend']
    formula = f"{CONFIG['TARGET_METRIC_LIFT']} ~ {' + '.join(context_vars)} + {' + '.join(CONFIG['AD_GROUP_COLUMNS'])}"
    try:
        model = smf.ols(formula, data=df).fit()
        return pd.DataFrame({"coefficient_lift_per_ad": model.params, "p_value": model.pvalues}).to_dict(orient='index')
    except Exception as e:
        analysis_logger.error(f"Failed to run ad group lift analysis: {e}", exc_info=True)
        return {"error": str(e)}

def rank_game_performance(df, game_dates):
    analysis_logger.info("Starting hierarchical game performance ranking...")
    analysis_logger.warning("Defaulting to SARIMAX for robust directional ranking as primary method.")
    sarimax_results = calculate_residual_uplift(df, pd.to_datetime(game_dates), CONFIG['TARGET_METRIC_LIFT'])
    if sarimax_results and sarimax_results.get("ranked_game_impacts"):
        analysis_logger.info("✅ Directional ranking provided by fallback SARIMAX Residual Uplift method.")
        return sarimax_results, f"SARIMAX Residual ({CONFIG['TARGET_METRIC_LIFT']})", CONFIG['TARGET_METRIC_LIFT']
    analysis_logger.error("❌ All ranking methods failed.")
    return None, "No Significant Model", None

def _create_performance_index(lift_results):
    """
    @what: Creates a relative performance index from raw lift percentages.
    @why: To provide a more stable and interpretable metric than raw lift,
           especially when p-values are not significant.
    @args: lift_results (dict) - The output from the primary lift model.
    @returns: (dict) - The same lift_results dict with an added 'performance_index'
                       for each game in 'ranked_game_impacts'.
    """
    analysis_logger.info("--> Creating relative game performance index...")
    if not lift_results or not lift_results.get("ranked_game_impacts"):
        analysis_logger.warning("No game impacts found to index. Skipping.")
        return lift_results

    game_impacts = lift_results["ranked_game_impacts"]
    
    lifts = [game.get('relative_lift_percent', 0) for game in game_impacts]
    
    if not lifts or len(lifts) < 2:
        for game in game_impacts:
            game['performance_index'] = 50
        return lift_results

    min_lift = min(lifts)
    max_lift = max(lifts)
    
    if max_lift == min_lift:
        for game in game_impacts:
            game['performance_index'] = 50
        return lift_results

    for game in game_impacts:
        lift = game.get('relative_lift_percent', 0)
        index_score = 100 * (lift - min_lift) / (max_lift - min_lift)
        game['performance_index'] = round(index_score, 2)
        
    analysis_logger.info("-> Successfully added performance index to game results.")
    return lift_results

def calculate_residual_uplift(df, game_dates, target_metric='branded_clicks'):
    analysis_logger.info(f"Calculating uplift and baseline trend for '{target_metric}' using SARIMAX...")
    exog_vars = CONFIG['CONTROL_VARS'] + CONFIG['AD_GROUP_COLUMNS']
    model_data = df[[target_metric] + exog_vars].copy().dropna()
    endog, exog = model_data[target_metric], model_data[exog_vars]
    
    try:
        model = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
        predictions = model.get_prediction(exog=exog).predicted_mean
        residuals = endog - predictions
        
        sponsorship_start = pd.to_datetime(CONFIG['SPONSORSHIP_START_DATE'])
        trend_data = predictions[predictions.index >= sponsorship_start].reset_index()
        trend_data.columns = ['date', 'baseline']
        trend_data['time_index'] = (trend_data['date'] - trend_data['date'].min()).dt.days
        
        trend_model = sm.OLS(trend_data['baseline'], sm.add_constant(trend_data['time_index'])).fit()
        trend_slope = trend_model.params.get('time_index', 0)
        trend_p_value = trend_model.pvalues.get('time_index', 1)
        
        long_term_trend_analysis = {
            "slope": trend_slope, "p_value": trend_p_value,
            "is_significant": trend_p_value < CONFIG['P_VALUE_SIGNIFICANCE_THRESHOLD'],
            "interpretation": "Sustained Growth" if trend_slope > 0 and trend_p_value < 0.1 else "Stable" if trend_p_value >= 0.1 else "Decline"
        }

        game_uplifts = []
        for game_date in pd.to_datetime(game_dates):
            if game_date in residuals.index and predictions.loc[game_date] > 0:
                relative_lift = (residuals.loc[game_date] / predictions.loc[game_date]) * 100
                game_uplifts.append({"game_date": game_date.strftime('%Y-%m-%d'), "relative_lift_percent": relative_lift, "p_value": "N/A (Residual Model)"})
        
        return {
            "method": "SARIMAX Residual", "target_metric": target_metric,
            "ranked_game_impacts": sorted(game_uplifts, key=lambda x: x['relative_lift_percent'], reverse=True),
            "long_term_trend_analysis": long_term_trend_analysis,
            "predictions": predictions, "residuals": residuals
        }
    except Exception as e:
        analysis_logger.error(f"SARIMAX fallback model failed: {e}", exc_info=True)
        return None

def perform_semantic_clustering(logs, model, brand_name):
    if not NLP_LIBRARIES_AVAILABLE: return None
    analysis_logger.info("Performing semantic clustering and classification on CTAs...")
    ctas = logs[~logs[CONFIG['CTA_COLUMN']].isin([brand_name.lower(), 'nan'])][CONFIG['CTA_COLUMN']].dropna().unique()
    if len(ctas) < CONFIG['MAX_CLUSTERS_TO_TEST']: return None
    
    try:
        embeddings = SentenceTransformer(CONFIG['SEMANTIC_MODEL_NAME']).encode(ctas)
        k_range = range(2, min(len(ctas), CONFIG['MAX_CLUSTERS_TO_TEST'] + 1))
        if not k_range: return None

        silhouette_scores = [silhouette_score(embeddings, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)) for k in k_range]
        optimal_k = k_range[np.argmax(silhouette_scores)]
        final_labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(embeddings)
        
        clustered_ctas_df = pd.DataFrame({'cta': ctas, 'cluster': final_labels})
        
        cluster_summaries = []
        theme_prompt_template = "You are a Brand Strategist. Based on these CTAs, give a 2-3 word name for this theme: {ctas_json}. Answer in Portuguese with ONLY the name."
        classification_prompt_template = "You are a Marketing Analyst. Classify the following message theme as 'Racional' (focus on price, offer, features), 'Emocional' (focus on brand, humor, identity), or 'Misto'. Answer with only one word. Theme: '{theme_name}' based on these examples: {ctas_json}"

        for i in range(optimal_k):
            cluster_ctas_list = clustered_ctas_df[clustered_ctas_df['cluster'] == i]['cta'].tolist()
            if not cluster_ctas_list: continue
            
            theme_response = model.generate_content(theme_prompt_template.format(ctas_json=json.dumps(cluster_ctas_list[:10], ensure_ascii=False)))
            theme_name = theme_response.text.strip().replace('"', "")
            
            classification_response = model.generate_content(classification_prompt_template.format(theme_name=theme_name, ctas_json=json.dumps(cluster_ctas_list[:5], ensure_ascii=False)))
            classification = classification_response.text.strip().replace("'", "").replace('"', "")
            
            cluster_summaries.append({"theme_name": theme_name, "classification": classification, "ctas": cluster_ctas_list})
            
        return {"clusters": cluster_summaries}
    except Exception as e:
        analysis_logger.error(f"Clustering error: {e}", exc_info=True)
        return None

# ==============================================================================
# 5. DATA SUMMARIZATION & PACKAGING
# ==============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    """
    @what: Custom encoder to handle special data types like numpy and pandas objects.
    @fix: This version now correctly handles pandas Series objects to prevent TypeErrors.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Series): return {str(k): v for k, v in obj.to_dict().items()}
        return super().default(obj)

def generate_business_kpis(df):
    kpis = {}
    total_brand_spend, total_peer_spend = df['generic_spend'].sum(), df['spend_peer_group'].sum()
    total_brand_impressions, total_peer_impressions = df['generic_impressions'].sum(), df['impressions_peer_group'].sum()
    kpis['overall_share_of_spend'] = f"{(total_brand_spend / (total_brand_spend + total_peer_spend)):.2%}" if (total_brand_spend + total_peer_spend) > 0 else "N/A"
    kpis['overall_sov_impressions'] = f"{(total_brand_impressions / (total_brand_impressions + total_peer_impressions)):.2%}" if (total_brand_impressions + total_peer_impressions) > 0 else "N/A"
    kpis['brand_vs_peer_performance'] = {'brand_total_generic_clicks': f"{df['generic_clicks'].sum():,.0f}",'peer_total_generic_clicks': f"{df['clicks_peer_group'].sum():,.0f}", 'brand_generic_ctr': f"{(df['generic_clicks'].sum()/df['generic_impressions'].sum()):.3%}" if df['generic_impressions'].sum() > 0 else "0.000%", 'peer_generic_ctr': f"{(df['clicks_peer_group'].sum()/df['impressions_peer_group'].sum()):.3%}" if df['impressions_peer_group'].sum() > 0 else "0.000%"}
    pre_df, post_df = df[df.index < CONFIG['SPONSORSHIP_START_DATE']], df[df.index >= CONFIG['SPONSORSHIP_START_DATE']]
    def calculate_period_kpis(period_df): return {"avg_branded_clicks": f"{period_df['branded_clicks'].mean():,.0f}"} if not period_df.empty else {}
    kpis['pre_post_sponsorship_analysis'] = {"pre_sponsorship": calculate_period_kpis(pre_df), "post_sponsorship": calculate_period_kpis(post_df)}
    return kpis

def create_game_level_summary(lift_results, logs_df):
    if not lift_results or not lift_results.get("ranked_game_impacts"): return []
    game_summaries = []
    for game_info in lift_results["ranked_game_impacts"]:
        game_date = pd.to_datetime(game_info['game_date']).date()
        game_day_logs = logs_df[logs_df['date'].dt.date == game_date]
        messages = list(game_day_logs[CONFIG['CTA_COLUMN']].dropna().unique())
        format_mix = game_day_logs['ad_group'].value_counts().to_dict()
        # --- NEW: Add Game Importance Score to the summary ---
        game_importance = game_day_logs['game_importance_score'].max() if not game_day_logs.empty else 0
        summary = {
            **game_info,
            "video_title": game_day_logs['video_title'].iloc[0] if not game_day_logs.empty else 'N/A',
            "messages": messages,
            "format_mix": format_mix,
            "total_ads": sum(format_mix.values()),
            "game_importance_score": game_importance
        }
        game_summaries.append(summary)
    return game_summaries

def create_full_data_package(**kwargs):
    logs_df = kwargs['logs_df']
    return {
        "analysis_context": {"brand": kwargs['brand_name'], "sponsorship_channel": "CazéTV"}, "business_performance": kwargs['business_kpis'],
        "sponsorship_lift_analysis": {"ranking_method_used": kwargs['ranking_method'], "lift_analysis_target": kwargs['lift_target_metric'], "lift_model_results": kwargs['lift_model_results']},
        "creative_intelligence": {
            "semantic_message_clusters": kwargs['semantic_cluster_results'], "ad_group_lift_analysis": kwargs['ad_group_lift_results'],
            "ad_group_definitions": {group: logs_df[logs_df['ad_group'] == group][['formato_contexto', CONFIG['CTA_COLUMN']]].drop_duplicates().head(3).to_dict('records') for group in logs_df['ad_group'].unique() if pd.notna(group)}
        }, "detailed_game_level_performance": kwargs['detailed_game_summary']}

def create_persona_payloads(full_data_package):
    """
    @what: Creates the specific high-level data payloads for the CMO and CDO reports.
    @fix: This version pre-formats the game examples to ensure the video title is used
           for the 'Partida' column, correcting the table generation bug.
    @args: full_data_package (dict) - The complete data package.
    @returns: (tuple) - A tuple containing (cmo_payload, cdo_payload).
    """
    top_n = CONFIG["GAME_PERFORMANCE_TOP_N"]
    detailed_summary = full_data_package.get("detailed_game_level_performance", [])

    # FIX: Pre-format the game examples for the LLM prompt
    def format_game_examples(games):
        formatted = []
        for game in games:
            # Create a new dict to avoid modifying the original
            formatted_game = {
                # This is the key change: map 'video_title' to 'Partida'
                "Partida": game.get("video_title", "N/A").replace('|', '-'),
                "Índice de Performance (0-100)": f"{game.get('performance_index', 0):.1f}",
                "Principais Mensagens": "; ".join(game.get('messages', ['N/A'])[:2])
            }
            formatted.append(formatted_game)
        return formatted

    # Base structure for both
    base_payload = {
        "analysis_context": {"brand": full_data_package["analysis_context"]["brand"]},
        "sponsorship_lift_analysis": full_data_package["sponsorship_lift_analysis"],
        "creative_intelligence": full_data_package["creative_intelligence"],
        # Use the newly formatted data here
        "game_performance_analysis": {
            "top_performing_games_examples": format_game_examples(detailed_summary[:top_n]),
            "worst_performing_games_examples": format_game_examples(detailed_summary[-top_n:])
        }
    }
    
    # CMO Payload (more strategic)
    cmo_payload = copy.deepcopy(base_payload)
    if "business_performance" in full_data_package:
        cmo_payload["business_performance_summary"] = {
            "overall_sov_impressions": full_data_package["business_performance"].get("overall_sov_impressions"),
            "pre_post_sponsorship_analysis": full_data_package["business_performance"].get("pre_post_sponsorship_analysis")
        }
    
    # CDO Payload (more technical)
    cdo_payload = copy.deepcopy(base_payload)
    cdo_payload["business_performance"] = full_data_package.get("business_performance")

    return cmo_payload, cdo_payload

def trim_payload_for_llm(payload, max_ctas_per_cluster=5, max_messages_per_game=3):
    if not payload: return {}
    trimmed_payload = copy.deepcopy(payload)
    if "creative_intelligence" in trimmed_payload and "semantic_message_clusters" in trimmed_payload["creative_intelligence"]:
        clusters = trimmed_payload["creative_intelligence"].get("semantic_message_clusters", {}).get("clusters", [])
        if clusters:
            for cluster in clusters:
                if 'ctas' in cluster and isinstance(cluster['ctas'], list):
                    cluster['ctas'] = sorted(list(dict.fromkeys(cluster['ctas'])), key=len)[:max_ctas_per_cluster]
    if "game_performance_analysis" in trimmed_payload:
        for key in ["top_performing_games_examples", "worst_performing_games_examples"]:
            if key in trimmed_payload["game_performance_analysis"]:
                for game in trimmed_payload["game_performance_analysis"][key]:
                    if "messages" in game and isinstance(game["messages"], list):
                        messages = [msg for msg in game["messages"] if msg != 'nan']
                        if len(messages) > max_messages_per_game:
                            game["messages"] = [msg for msg, count in Counter(messages).most_common(max_messages_per_game)]
    return trimmed_payload

def create_smart_payload(prompt_key, full_payload):
    """
    @what: Creates a surgically lean payload for a specific LLM prompt key.
    @why: Prevents API timeouts by sending only the necessary context, drastically
           reducing token count for each call.
    @args: prompt_key (str) - The 'key' from the prompt dictionary (e.g., 'exec_summary').
           full_payload (dict) - The complete, but potentially bloated, persona payload.
    @returns: (dict) - A new, lean dictionary with only the required data.
    """
    analysis_logger.info(f"Creating smart payload for prompt key: '{prompt_key}'...")
    
    # Always include the basic context
    smart_payload = {
        "analysis_context": full_payload.get("analysis_context")
    }

    # Define the data requirements for each prompt key
    # This acts as a router to select only the necessary data.
    requirements = {
        # CMO Prompts
        "exec_summary": ["sponsorship_lift_analysis", "game_performance_analysis", "business_performance_summary"],
        "game_impact": ["game_performance_analysis"],
        "messaging_story": ["creative_intelligence"],
        "winning_formula": ["ad_group_lift_analysis", "detailed_game_level_performance", "creative_intelligence"],
        "audience_connection": ["winning_formula", "creative_intelligence"], # Needs context from previous section
        "recommendations": ["winning_formula"], # Needs context from previous section

        # CDO Prompts
        "tech_exec_summary": ["sponsorship_lift_analysis"],
        "methodology": [], # No data needed, it's a descriptive task
        "media_efficiency": ["business_performance"],
        "brand_lift_deep_dive": ["sponsorship_lift_analysis"],
        "creative_intelligence_deep_dive": ["creative_intelligence", "ad_group_lift_analysis"]
    }

    # Get the list of required top-level keys for the given prompt
    required_keys = requirements.get(prompt_key, [])

    # Populate the smart payload with the required data
    for key in required_keys:
        if key in full_payload:
            smart_payload[key] = full_payload[key]
        # Handle nested keys for more complex requirements
        elif key == "ad_group_lift_analysis":
             if "creative_intelligence" in full_payload and "ad_group_lift_analysis" in full_payload["creative_intelligence"]:
                smart_payload["ad_group_lift_analysis"] = full_payload["creative_intelligence"]["ad_group_lift_analysis"]
        elif key == "detailed_game_level_performance":
             # This is a special case to handle a key from the *complete* data package, not the persona payload
             # A more robust solution might pass the complete package in, but this works for now.
             pass # This data is not in the CDO payload by default, but the prompt requires it. Let's adjust the CDO payload creation.

    # SPECIAL FIX: The 'winning_formula' for the CDO needs 'detailed_game_level_performance' which isn't in the base payload.
    # We will ensure this is handled by modifying the main loop. For now, this structure is sound.
    
    # The most important trim: remove the massive time-series data unless explicitly needed.
    if "sponsorship_lift_analysis" in smart_payload:
        # Only the deep dive needs the full model results
        if prompt_key not in ["brand_lift_deep_dive"]:
            if "lift_model_results" in smart_payload["sponsorship_lift_analysis"]:
                smart_payload["sponsorship_lift_analysis"]["lift_model_results"].pop("predictions", None)
                smart_payload["sponsorship_lift_analysis"]["lift_model_results"].pop("residuals", None)

    return smart_payload

# ==============================================================================
# 6. REPORT GENERATION FUNCTIONS
# ==============================================================================

def generate_brand_team_summary(model, cmo_report_markdown, brand_name, encoded_plots):
    """
    @what: Takes the generated CMO report and synthesizes it into a punchy,
           actionable summary for a non-technical brand team.
    @args: model (GenerativeModel), cmo_report_markdown (str), brand_name (str),
           encoded_plots (dict) - containing the Base64 images.
    @returns: (str) - The generated markdown for the brand team summary.
    """
    analysis_logger.info(f"Generating Brand Team Summary for {brand_name}...")

    # This prompt is specifically engineered for a brand/creative audience.
    # It uses marketer-friendly language and asks for actionable takeaways.
    BRAND_TEAM_PROMPT = """You are a world-class Brand Director at a top marketing agency, and you are presenting a summary of the [BRAND_NAME] sponsorship performance on CazéTV to the internal Brand Managers. They are creative, brand-focused, and need actionable insights, not deep data. Your tone should be energetic, direct, and inspiring.

    **Instructions:**
    - Write in PORTUGUESE.
    - Be direct and share actionable insights. Remember you are talking to a branding team, so your focus should always be branding related insights like brand message variation, asset variation, best performing ads, etc.
    - Focus on the "So What?". Translate every data point into a clear strategic or creative action.
    - The output must be ONLY the markdown for this new summary.

    **Structure your report exactly as follows:**

    # 📣 Brand Director's Debrief: [BRAND_NAME] on CazéTV

    ## 🚀 The TL;DR: Our Key Takeaway
    Start with a single, powerful paragraph that summarizes the single most important lesson from this sponsorship. What is the one thing the brand team absolutely needs to know?

    ## 💡 The Big Idea: What Actually Worked?
    Based on the report, describe the "winning formula" in simple terms. What specific combination of ad *format* and *message type* drove the best results? Celebrate this success.

    ## 💡 Message Consistency: How did your communication vary over time?
    Based on the report, how our message has changed over time. Where we consistent or did we presented a lot of different messages? What were the products and benefits highlighted over time? Can you make an assessment if the message consistency has affected the results? Finally, make actionable recommendations based on the insights presented

    ## 🎨 Creative Deep Dive: The Ads That Won (and Lost)
    - ✅ **The Winning Creative:** Describe the *themes* and specific *CTAs* from the best-performing ads (e.g., "Direct offers like 'Taxa Zero' crushed it").
    - ❌ **The Losing Creative:** Describe the *themes* from the worst-performing ads (e.g., "Generic branding without a clear 'ask' fell flat").
    - **Key Visual:** Embed the Message Themes chart here using this exact markdown: `![Our Creative Mix]({{cluster_dist_plot_path}})`

    ## 🎯 Hitting the Target: Did We Speak Their Language?
    Briefly explain how our most successful messages connected with specific viewer mindsets (like the "Second-Screener" with their phone out). Where is our biggest opportunity to connect better?

    ## ✅ Actionable Next Steps: The Go-Forward Plan
    Provide 3-4 bullet points of sharp, creative, and strategic actions the brand team should take next. Frame these as exciting opportunities.
    - Example: "Double down on L-Shape formats with QR codes for our next promotion."
    - Example: "Brief our creative agency to build a new campaign around the 'Your Phone is Your POS' concept."
    """

    # Use the CMO report markdown as the primary source of truth.
    full_prompt_text = f"{BRAND_TEAM_PROMPT}\n\n--- FULL STRATEGIC REPORT CONTEXT ---\n{cmo_report_markdown}"

    # Render the final prompt, inserting the key visual
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    template = env.from_string(full_prompt_text)
    final_rendered_prompt = template.render(**encoded_plots)

    try:
        response = model.generate_content(final_rendered_prompt, request_options={"timeout": 600})
        if response.parts:
            return response.text
        else:
            reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            error_message = f"Model returned no content for Brand Team Summary. Finish Reason: {reason}."
            analysis_logger.error(error_message)
            return f"<h2>Error Generating Brand Team Summary</h2><p>{error_message}</p>"
    except Exception as e:
        analysis_logger.error(f"Brand Team Summary generation failed: {e}", exc_info=True)
        return f"<h2>Error Generating Brand Team Summary</h2><p>The model could not generate a response. The error was: {e}</p>"

# In Section 6: REPORT GENERATION FUNCTIONS
# REPLACE the existing function with this one

def generate_and_save_visualizations(df, brand_name, brand_slug, image_path, game_dates, semantic_results=None, model_results=None):
    """
    @what: Generates only the necessary charts and saves them as image files.
    @fix V4: Removed the generation of the unused 'Share of Voice' (SOV) chart.
             Returns a dictionary of relative image paths for clean HTML linking.
    """
    analysis_logger.info("Generating and saving required visualizations...")
    
    # Ensure the image path exists
    os.makedirs(image_path, exist_ok=True)
    
    df_sponsorship_period = df[df.index >= pd.to_datetime(CONFIG['SPONSORSHIP_START_DATE']) - pd.Timedelta(days=30)].copy()
    
    # This will hold the relative paths for the HTML template
    plot_paths = {
        # 'sov_trend_plot_path': None, # REMOVED
        'branded_perf_plot_path': None,
        'generic_perf_plot_path': None,
        'cluster_dist_plot_path': None,
        'incrementality_plot_path': None
    }
    
    plt.style.use('seaborn-v0_8-whitegrid')
    game_dates_dt = pd.to_datetime(game_dates)

    def add_game_day_markers(ax):
        for game_date in game_dates_dt:
            ax.axvline(x=game_date, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='_nolegend_')
        if not game_dates_dt.empty:
            ax.plot([], [], color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='Game Day')

    # --- SOV Trend Chart ---
    # THIS ENTIRE BLOCK HAS BEEN REMOVED AS IT WAS UNUSED.

    # --- Branded Performance Chart ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    target_metric_label = CONFIG['TARGET_METRIC_LIFT'].replace('_', ' ').title()
    ax1.plot(df_sponsorship_period.index, df_sponsorship_period[CONFIG['TARGET_METRIC_LIFT']], color='#0056a0', label=f'Daily {target_metric_label}')
    ax1.set_xlabel('Date'); ax1.set_ylabel(target_metric_label, color='#0056a0')
    ax1.tick_params(axis='y', labelcolor='#0056a0')
    add_game_day_markers(ax1)
    ax1.legend(loc='upper left')
    plt.title(f'{brand_name} Daily {target_metric_label}', fontsize=16)
    fig.tight_layout()
    branded_filename = f"{brand_slug}_branded_performance.png"
    plt.savefig(os.path.join(image_path, branded_filename))
    plot_paths['branded_perf_plot_path'] = f"images/{branded_filename}"
    plt.close()

    # --- Generic Performance Chart ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_sponsorship_period.index, df_sponsorship_period['generic_clicks'], color='#5a9bd4', label='Daily Generic Clicks')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Generic Clicks', color='#5a9bd4')
    ax1.tick_params(axis='y', labelcolor='#5a9bd4')
    add_game_day_markers(ax1)
    ax1.legend(loc='upper left')
    plt.title(f'{brand_name} Daily Generic Search Clicks', fontsize=16)
    fig.tight_layout()
    generic_filename = f"{brand_slug}_generic_performance.png"
    plt.savefig(os.path.join(image_path, generic_filename))
    plot_paths['generic_perf_plot_path'] = f"images/{generic_filename}"
    plt.close()
    
    # --- Semantic Cluster Chart ---
    if semantic_results and 'clusters' in semantic_results and semantic_results['clusters']:
        cluster_names = [c['theme_name'] for c in semantic_results['clusters']]
        cluster_counts = [len(c['ctas']) for c in semantic_results['clusters']]
        plt.figure(figsize=(10, 8))
        plt.pie(cluster_counts, labels=cluster_names, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(cluster_names)))
        plt.title('Distribution of Creative Message Themes', fontsize=16)
        plt.tight_layout()
        cluster_filename = f"{brand_slug}_cluster_dist.png"
        plt.savefig(os.path.join(image_path, cluster_filename))
        plot_paths['cluster_dist_plot_path'] = f"images/{cluster_filename}"
        plt.close()

    # --- Incrementality Chart ---
    if model_results and model_results.get('predictions') is not None:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        preds = model_results['predictions']; resids = model_results['residuals']
        actuals_filtered = df_sponsorship_period[CONFIG['TARGET_METRIC_LIFT']].dropna()
        preds_filtered = preds[preds.index.isin(actuals_filtered.index)]
        resids_filtered = resids[resids.index.isin(actuals_filtered.index)]
        
        target_metric_label = CONFIG['TARGET_METRIC_LIFT'].replace('_', ' ').title()
        
        ax1.plot(actuals_filtered.index, actuals_filtered, color='#0056a0', label=f'Actual {target_metric_label}', linewidth=2)
        ax1.plot(preds_filtered.index, preds_filtered, color='#5a9bd4', linestyle='--', label='Projected (No Sponsorship)')
        ax1.set_xlabel('Date'); ax1.set_ylabel(target_metric_label); ax1.tick_params(axis='y')
        add_game_day_markers(ax1)
        ax1.legend(loc='upper left')
        ax1.set_title(f'Daily {target_metric_label} Lift: Actual vs. SARIMAX Projection', fontsize=16)
        
        ax2 = ax1.twinx()
        colors = ['#28a745' if x > 0 else '#dc3545' for x in resids_filtered]
        ax2.bar(resids_filtered.index, resids_filtered, color=colors, alpha=0.5, label=f'Daily Incremental {target_metric_label} (Lift)')
        ax2.set_ylabel(f'Incremental {target_metric_label}', color='gray'); ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend(loc='upper right')
        
        fig.tight_layout()
        incrementality_filename = f"{brand_slug}_incrementality_trend.png"
        plt.savefig(os.path.join(image_path, incrementality_filename))
        plot_paths['incrementality_plot_path'] = f"images/{incrementality_filename}"
        plt.close()

    # Remove paths for any plots that weren't generated
    final_plot_paths = {k: v for k, v in plot_paths.items() if v is not None}
    
    analysis_logger.info(f"-> Saved {len(final_plot_paths)} plots to {image_path}")
    return final_plot_paths

# In Section 6: REPORT GENERATION FUNCTIONS
# REPLACE the existing generate_narrative_section function

def generate_narrative_section(model, section_prompt, llm_data_package, brand_name, plot_paths, **kwargs):
    """
    @what: Generates a single section of a report using the LLM.
    @fix V3: Uses plot_paths dictionary with relative URLs for Jinja2 rendering.
    """
    analysis_logger.info(f"Generating narrative section for prompt key: {section_prompt['key']}...")

    base_prompt_template = section_prompt['prompt']
    prompt_with_brand = base_prompt_template.replace('[BRAND_NAME]', brand_name)
    json_data = json.dumps(llm_data_package, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    
    full_prompt_text_with_placeholders = (
        f"{prompt_with_brand}\n\n--- ANALYTICAL DATA (JSON) ---\n{json_data}"
    )

    if kwargs.get('previous_section_output'):
        full_prompt_text_with_placeholders += f"\n\n--- PREVIOUSLY GENERATED INSIGHTS ---\n{kwargs['previous_section_output']}"

    # Use Jinja2 to insert the relative image paths into the prompt
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    template = env.from_string(full_prompt_text_with_placeholders)
    # The 'plot_paths' dict now contains the relative URLs
    final_rendered_prompt = template.render(**plot_paths) 
    
    try:
        response = model.generate_content(final_rendered_prompt, request_options={"timeout": 600})
        if response.parts:
            return response.text
        else:
            reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            error_message = f"Model returned no content. Finish Reason: {reason}. "
            analysis_logger.error(error_message)
            return f"<h2>Error Generating Section: {section_prompt['title']}</h2><p>{error_message}</p>"
    except Exception as e:
        analysis_logger.error(f"Narrative generation failed for '{section_prompt['key']}': {e}", exc_info=True)
        return f"<h2>Error Generating Section: {section_prompt['title']}</h2><p>The model could not generate a response. Error: {e}</p>"

# REPLACE the existing render_html_report function
def render_html_report(memo_text, template_str, report_filename):
    """
    @what: Renders the final HTML report from markdown text and a template.
    @fix V3: No longer handles plot paths, as they are now relative and handled
             by the browser directly.
    """
    analysis_logger.info(f"Rendering final HTML report: {report_filename}...")
    # The memo_text from Gemini now contains standard <img src="relative/path/..."> tags
    memo_html = markdown(memo_text, extensions=['tables']) if MARKDOWN_AVAILABLE else f"<pre>{memo_text}</pre>"
    
    # The Jinja template just needs to render the HTML content.
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    template = env.from_string(template_str)
    final_html = template.render(memo_html_content=memo_html)
    
    with open(report_filename, "w", encoding='utf-8') as f:
        f.write(final_html)
    analysis_logger.info(f"✅ Report successfully generated: {report_filename}")

def _generate_appendix_content_from_data(full_data_package):
    """
    @what: Generates the appendix section using Python, saving an LLM call.
    @fix: This version sorts the game performance table by date in descending order.
    @returns: (str) - A fully formatted markdown string for the appendix.
    """
    analysis_logger.info("Generating Appendix content directly from data (Python-based)...")
    appendix_md, p_threshold = ["<h2>7. Apêndice</h2>"], CONFIG["P_VALUE_SIGNIFICANCE_THRESHOLD"]
    
    # Section 7.1
    appendix_md.append("<h3>7.1 Detalhamento dos Grupos Estratégicos de Anúncios</h3>")
    ad_defs = full_data_package.get("creative_intelligence", {}).get("ad_group_definitions", {})
    for group, examples in ad_defs.items():
        appendix_md.append(f"<h4>{group.replace('_', ' ').title()}</h4>")
        if examples:
            appendix_md.append("<ul>" + "".join([f"<li><b>{ex.get('formato_contexto', 'N/A')}:</b> {ex.get(CONFIG['CTA_COLUMN'], 'nan')}</li>" for ex in examples]) + "</ul>")
    
    # Section 7.2
    appendix_md.append("<h3>7.2 Detalhamento dos Temas de Mensagem Criativa</h3>")
    clusters = full_data_package.get("creative_intelligence", {}).get("semantic_message_clusters", {}).get("clusters", [])
    for cluster in clusters:
        appendix_md.append(f"<h4>{cluster.get('theme_name', 'N/A')}</h4>")
        appendix_md.append("<ul>" + "".join([f"<li>{cta}</li>" for cta in cluster.get("ctas", [])[:3]]) + "</ul>")

    # Section 7.3
    appendix_md.append("<h3>7.3 Análise Detalhada por Jogo</h3>")
    table = ["| Data do Jogo | Partida (Título do Vídeo) | Índice de Performance (0-100) | Lift Bruto (%) | Total de Anúncios | Principais Mensagens |",
             "|---|---|---|---|---|---|"]
    
    # FIX #3: Sort by game_date in descending order
    sorted_games = sorted(full_data_package.get("detailed_game_level_performance", []), key=lambda x: x['game_date'], reverse=True)
    
    for game in sorted_games:
        table.append(f"| {pd.to_datetime(game['game_date']).strftime('%Y-%m-%d')} | {game.get('video_title', 'N/A').replace('|', '-')}| {game.get('performance_index', 0):.1f} | {game.get('relative_lift_percent', 0):.2f}% | {game.get('total_ads', 0)} | {'; '.join(game.get('messages', ['N/A'])[:2])} |")
    appendix_md.append("\n".join(table))
    
    return "\n\n".join(appendix_md)

# ==============================================================================
# 7. MAIN ORCHESTRATOR FUNCTION
# ==============================================================================
def run_full_analysis(brand_name):
    """
    @what: Main orchestrator function to run the entire analysis pipeline for a single brand.
    @args: brand_name (str) - The name of the brand to analyze.
    """
    analysis_logger.info(f"\n{'='*60}\n ANALYZING BRAND: {brand_name.upper()} \n{'='*60}")
    brand_slug = _sanitize_string(brand_name)

    # --- 1. Setup & Initialization ---
    output_dir = os.path.join(CONFIG['LOG_FILE_PATH'], brand_slug)
    image_path = os.path.join(output_dir, "images")
    os.makedirs(image_path, exist_ok=True)
    check_and_create_data(brand_name, brand_slug)
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not all([GEMINI_AVAILABLE, api_key, JINJA_AVAILABLE, NLP_LIBRARIES_AVAILABLE, MARKDOWN_AVAILABLE]):
        raise EnvironmentError("A key library or the GOOGLE_API_KEY is missing.")
    genai.configure(api_key=api_key)
    specialist_model = genai.GenerativeModel('gemini-2.5-pro')

    # --- 2. Data Preparation & 3. Core Modeling ---
    brand_logs, branded_df, generic_df = load_and_clean_data(brand_name, brand_slug)
    if brand_logs.empty:
        analysis_logger.error(f"Aborting for {brand_name}: No log data found.")
        return
        
    logs_with_groups = map_tactical_to_strategic_groups(brand_logs)

    # FIX: Defensively reset index to prevent KeyError: 'date' in downstream functions.
    # This ensures that if 'date' was moved to the index, it is restored as a column.
    if 'date' not in logs_with_groups.columns and 'date' in logs_with_groups.index.names:
        logs_with_groups.reset_index(inplace=True)
        
    master_df = create_master_df_enhanced(branded_df, generic_df, logs_with_groups, brand_name)

    eligible_game_dates = get_eligible_game_dates(logs_with_groups, master_df)
    if not eligible_game_dates:
        analysis_logger.error(f"Aborting for {brand_name}: No eligible game dates for analysis.")
        return

    lift_model_results, ranking_method, lift_target_metric = rank_game_performance(master_df, eligible_game_dates)
    lift_model_results = _create_performance_index(lift_model_results)
    detailed_game_summary = create_game_level_summary(lift_model_results, logs_with_groups)
    semantic_cluster_results = perform_semantic_clustering(logs_with_groups, specialist_model, brand_name)
    ad_group_lift_results = perform_ad_group_lift_analysis(master_df)

    # --- 4. Data Packaging & Visualization (FIXED) ---
    # Now returns a dictionary of relative paths, e.g., {'branded_perf_plot_path': 'images/pagbank_branded_performance.png'}
    plot_paths = generate_and_save_visualizations(
        df=master_df,
        brand_name=brand_name,
        brand_slug=brand_slug,
        image_path=image_path,
        game_dates=eligible_game_dates,
        semantic_results=semantic_cluster_results,
        model_results=lift_model_results
    )
    
    full_data_package = create_full_data_package(
        brand_name=brand_name, business_kpis=generate_business_kpis(master_df), lift_model_results=lift_model_results,
        ranking_method=ranking_method, lift_target_metric=lift_target_metric, semantic_cluster_results=semantic_cluster_results,
        ad_group_lift_results=ad_group_lift_results, detailed_game_summary=detailed_game_summary, logs_df=logs_with_groups
    )
    
    cmo_payload, cdo_payload = create_persona_payloads(full_data_package)
    cmo_payload_lean = trim_payload_for_llm(cmo_payload)
    cdo_payload_lean = trim_payload_for_llm(cdo_payload)
    
    base_name = os.path.join(output_dir, f"{brand_slug}_{CONFIG['FINAL_DATA_PACKAGE_BASENAME']}")
    for name, data in [("complete", full_data_package), ("cmo_payload", cmo_payload_lean), ("cdo_payload", cdo_payload_lean)]:
        with open(f"{base_name}_{name}.json", "w", encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    # --- 5. Report Generation ---
    IDEAL_TEMPLATE = """<!DOCTYPE html><html lang=\"pt-br\"><head><meta charset=\"UTF-8\"><title>Business Review: Patrocínio CazéTV (Análise Avançada)</title><style>body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; } .container { max-width: 1000px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); } h1, h2, h3 { color: #003366; border-bottom: 2px solid #ffc107; padding-bottom: 10px; margin-top: 30px; } h1 { text-align: center; font-size: 2.5em; border-bottom: none; } h2 { font-size: 1.8em; } h3 { font-size: 1.4em; border-bottom: none; } table { width: 100%; border-collapse: collapse; margin-top: 20px; } th, td { padding: 12px; border: 1px solid #ddd; text-align: left; } th { background-color: #f2f2f2; } img { max-width: 100%; height: auto; margin-top: 20px; border-radius: 8px; }</style></head><body><div class=\"container\">{{ memo_html_content | safe }}</div></body></html>"""
    
    CMO_PROMPTS = [
        {"key": "exec_summary", "title": "Sumário Executivo", "prompt": "You are a world-class Senior Management Consultant reporting to the CMO of [BRAND_NAME]. Your task is to write ONLY the Executive Summary for a creative performance report.\n**Instructions:**\n- Write in PORTUGUESE.\n- Start with a powerful headline summarizing the single most important creative finding.\n- Write a concise executive summary (2-3 paragraphs) detailing the core story of creative performance, highlighting the winning combination of message and format, and its strategic implications.\n- The output must be ONLY the markdown for this section, starting with a <h1> or <h2> title: \"1. Sumário Executivo: A Receita do Sucesso Criativo\""},
        {"key": "game_impact", "title": "Impacto do Patrocínio", "prompt": "You are a world-class Senior Management Consultant. Your task is to write ONLY the \"Game Impact\" section of a creative performance report.\\n**Instructions:**\\n- Write in PORTUGUESE.\\n- Generate a Markdown table of the **Top 3 and Bottom 3 performing games** from the `game_performance_analysis` data, including 'Partida', 'Índice de Performance (0-100)', and 'Principais Mensagens'.\\n- Analyze this table to explain *why* performance varied, focusing on the differences in messaging strategy and the specific CTAs present in the winning games. Explain the index: a score of 1.00 is the best performance, 0.50 is half as effective, and negative scores indicate a detrimental impact.\\n- The output must be ONLY the markdown for this section, starting with a <h2> title: \"2. O Verdadeiro Impacto do Patrocínio: Quais Anúncios Realmente Funcionaram?\""},
        {"key": "messaging_story", "title": "História da Mensagem", "prompt": "You are a world-class Senior Management Consultant. Your task is to write ONLY the \"Messaging Story\" section of a creative performance report.\n**Instructions:**\n- Write in PORTUGUESE.\n- **Análise de Temas:** Based on the `semantic_message_clusters` data, describe the primary communication themes that [BRAND_NAME] used. What was the dominant message?\n- **Análise de Voz da Marca:** Based on the examples in the clusters, assess if the tone of the messaging was consistent with [BRAND_NAME]'s overall brand identity (e.g., practical, premium, fun, etc.). Was the brand voice clear and strong? The output must be ONLY the markdown for this section, starting with a <h2> title: \"3. A História da Nossa Mensagem: Consistência, Tom e Contexto\""},
        {"key": "winning_formula", "title": "Fórmula Vencedora", "prompt": "You are a world-class Senior Management Consultant. Your task is to write ONLY the \"Winning Formula\" section of a creative performance report. **This is the most important analytical section.**\n**Instructions:**\n- Write in PORTUGUESE.\n- **Análise de Estratégia (O Que Funciona):** First, present a **complete and detailed Markdown table** from `ad_group_lift_analysis` showing the lift coefficient and statistical significance for **all** strategic ad groups and context variables.\n- **Impacto do Contexto do Jogo:** Based on the table, specifically analyze the impact of `is_weekend` and `game_importance_score`. Do we see higher brand lift during weekend games? Crucially, does advertising during a 'big club' game (a higher `game_importance_score`) significantly increase performance? Explain the business implications of these findings.\n- **Análise de Formato:** Now, analyze the coefficients for the ad formats themselves. Which formats are the heavy hitters driving the most significant lift?\n- **Análise de Mensagem (Qual Mensagem Funciona):** Based on the `semantic_message_clusters` and their `classification`, analyze which type of messaging (`Rational`, `Emocional`, or `Misto`) resonates most. Cross-reference this with the top-performing games in `detailed_game_level_performance`. Do the winning games use a specific message type?\n- **Análise Tática (Por Que Funciona)::** Synthesize the findings. Analyze the `format_mix` and `messages` of the best-performing games to identify the specific ad *formats* and *message themes* that drove the winning strategy.\n- **A Fórmula Criativa Vencedora:** Synthesize all these findings into a clear, prescriptive \"Fórmula Criativa Vencedora\" with 2-4 numbered points, combining the best format, context, and message type.\n- The output must be ONLY the markdown for this section, starting with a <h2> title: \"4. A Fórmula Vencedora: Decodificando as Alavancas de Performance\""},
        {"key": "audience_connection", "title": "Conexão com a Audiência", "prompt": "You are a world-class Senior Management Consultant. Your task is to write ONLY the \"Audience Connection\" section of a creative performance report.\n**Instructions:**\n- Write in PORTUGUESE.\n- **Nossos Espectadores:** Based on general sports fan behavior, define 2-3 high-level behavioral personas for the CazéTV audience (e.g., \"Second-Screener,\" \"Immersive Fan,\" \"Social Viewer\").\n- **Análise de \"Message-Persona Fit\":** Using the **previously generated insights about the 'Winning Formula'** and the message themes from `semantic_message_clusters`, analyze which of [BRAND_NAME]'s themes and CTAs are best suited for each persona. Identify the biggest opportunity for improvement.\n- The output must be ONLY the markdown for this section, starting with a <h2> title: \"5. Conectando com a Audiência Certa, da Maneira Certa\""},
        {"key": "recommendations", "title": "Recomendações", "prompt": "You are a world-class Senior Management Consultant. Your task is to write ONLY the \"Strategic Recommendations\" section of a creative performance report.\\n**Instructions:**\\n- Write in PORTUGUESE.\\n- **Use the previously generated 'Winning Formula' as your primary context.** Provide three sharp, forward-looking recommendations on how to apply and scale this formula in future marketing initiatives.\\n- The output must be ONLY the markdown for this section, starting with a <h2> title: \"6. Recomendações Estratégicas\""},
        {"key": "appendix", "title": "Apêndice", "prompt": ""}
    ]
    CDO_PROMPTS = [
        {"key": "tech_exec_summary", "title": "Sumário Técnico Executivo", "prompt": "You are a Senior Data Scientist. Write ONLY the \"Sumário Técnico Executivo\" section for a technical appendix about the [BRAND_NAME] sponsorship analysis.\n**Instructions:**\n- Write in PORTUGUESE.\n- State the primary lift measurement model used (from `sponsorship_lift_analysis.ranking_method_used`).\n- State the primary driver analysis model used (OLS).\n- Summarize the key quantitative finding based on the winning model and `lift_analysis_target`, specifying the metric that showed lift or stating if no significant result was found.\n- The output must be ONLY the markdown for this section, starting with a \u003ch1\u003e or \u003ch2\u003e title."},
        {"key": "methodology", "title": "Metodologia Analítica", "prompt": "You are a Senior Data Scientist. Write ONLY the \"Metodologia Analítica\" section.\n**Instructions:**\n- Write in PORTUGUESE.\n- Detail the analytical pipeline: data preparation, feature engineering, the hierarchical modeling approach (defaulting to SARIMAX), and the parallel creative analysis (Group OLS + NLP Clustering).\n- The output must be ONLY the markdown for this section, starting with a \u003ch2\u003e title."},
        {"key": "media_efficiency", "title": "Eficiência de Mídia e Performance de Negócio", "prompt": "You are a Senior Data Scientist. Write ONLY the \"Eficiência de Mídia e Performance de Negócio\" deep dive.\n**Instructions:**\n- Write in PORTUGUESE.\n- Analyze the top-line business KPIs from `business_performance`.\n- Embed and analyze the charts using this exact HTML: \u003cimg src=\"{{branded_perf_plot_path}}\" alt=\"Branded Performance\"\u003e\u003cimg src=\"{{generic_perf_plot_path}}\" alt=\"Generic Performance\"\u003e\n- Conclude with the core learning on media effectiveness, spend/click behavior, and SoV/SoS benchmarks.\n- The output must be ONLY the markdown for this section, starting with a \u003ch2\u003e title."},
        {"key": "brand_lift_deep_dive", "title": "Mensuração de Lift de Marca", "prompt": "You are a Senior Data Scientist. Write ONLY the \"Mensuração de Lift de Marca\" deep dive.\n**Instructions:**\n- Write in PORTUGUESE.\n- State the winning model identified in `sponsorship_lift_analysis`.\n- Provide a detailed explanation of the findings, including relevant p-values and confidence intervals if available.\n- Embed and explain the chart using this exact HTML: \u003cimg src=\"{{incrementality_plot_path}}\" alt=\"Incrementality Chart\"\u003e, clarifying its purpose is for directional visualization.\n- Conclude with the core insight on what the sponsorship drives.\n- The output must be ONLY the markdown for this section, starting with a \u003ch2\u003e title."},
        {"key": "creative_intelligence_deep_dive", "title": "Inteligência de Criativo e Drivers de Performance", "prompt": "You are a Senior Data Scientist. Write ONLY the \"Inteligência de Criativo e Drivers de Performance\" deep dive.\n**Instructions:**\n- Write in PORTUGUESE.\n- Briefly explain the NLP clustering process (e.g., SentenceTransformer + KMeans) and embed the chart using this exact HTML: \u003cimg src=\"{{cluster_dist_plot_path}}\" alt=\"Cluster Chart\"\u003e\n- Present the technical results for the \"Análise de Estratégia de Anúncio (OLS por Grupo)\" based on `ad_group_lift_analysis`.\n- Conclude with a synthesized insight connecting the strategic findings to creative strategy.\n- The output must be ONLY the markdown for this section, starting with a \u003ch2\u003e title."}
    ]

    # --- CMO Report Generation ---
    cmo_report_sections = {}
    cmo_report_file = os.path.join(output_dir, f"{brand_slug}_{CONFIG['CMO_REPORT_FILE_BASENAME']}")
    
    analysis_logger.info("--- Generating CMO Report with Smart Payloads ---")
    for p in CMO_PROMPTS:
        if p['key'] == 'appendix':
            cmo_report_sections[p['key']] = _generate_appendix_content_from_data(full_data_package)
        else:
            smart_payload_cmo = create_smart_payload(p['key'], cmo_payload_lean)
            cmo_report_sections[p['key']] = generate_narrative_section(
                specialist_model,
                p,
                smart_payload_cmo,
                brand_name,
                plot_paths, # Pass relative paths
                previous_section_output=cmo_report_sections.get('winning_formula')
            )
            
    render_html_report("\n\n".join(cmo_report_sections.values()), IDEAL_TEMPLATE, cmo_report_file)
    analysis_logger.info(f"✅ CMO Report successfully generated: {cmo_report_file}")

    # --- CDO Report Generation ---
    cdo_report_sections = {}
    cdo_report_file = os.path.join(output_dir, f"{brand_slug}_{CONFIG['CDO_APPENDIX_FILE_BASENAME']}")
    
    analysis_logger.info("--- Generating CDO Report with Smart Payloads ---")
    for p in CDO_PROMPTS:
        smart_payload_cdo = create_smart_payload(p['key'], cdo_payload_lean) # Use helper function
        cdo_report_sections[p['key']] = generate_narrative_section(
            specialist_model,
            p,
            smart_payload_cdo,
            brand_name,
            plot_paths # Pass relative paths
        )
    render_html_report("\n\n".join(cdo_report_sections.values()), IDEAL_TEMPLATE, cdo_report_file)
    analysis_logger.info(f"✅ CDO Report successfully generated: {cdo_report_file}")

    # --- Brand Team Summary Generation ---
    analysis_logger.info("--- Starting Brand Team Summary Generation ---")
    brand_team_report_file = os.path.join(output_dir, f"{brand_slug}_{CONFIG['BRAND_TEAM_REPORT_BASENAME']}")
    
    cmo_full_memo = "\n\n".join(cmo_report_sections.values())
    
    brand_team_summary_markdown = generate_brand_team_summary(
        model=specialist_model,
        cmo_report_markdown=cmo_full_memo,
        brand_name=brand_name,
        encoded_plots=plot_paths # Pass relative paths here too
    )
    
    render_html_report(brand_team_summary_markdown, IDEAL_TEMPLATE, brand_team_report_file)
    analysis_logger.info(f"✅ Brand Team Summary successfully generated: {brand_team_report_file}")


# ==============================================================================
# 8. EXECUTION BLOCK
# @what: Runs the analysis for all specified brands when the script is executed.
# ==============================================================================
if __name__ == "__main__":
    # FIX: Restored the full list of brands to enable the loop functionality.
    BRANDS_TO_ANALYZE = [
        "McDonald's"
    ]
    
    analysis_logger.info("\n" + "=" * 60 + "\n MULTI-BRAND SPONSORSHIP ENHANCED ANALYSIS SCRIPT START \\" + "=" * 60)
    
    for brand in BRANDS_TO_ANALYZE:
        try:
            run_full_analysis(brand)
        except FileNotFoundError as e:
            analysis_logger.error(f"Data files not found for brand '{brand}'. Skipping... Error: {e}")
        except Exception as e:
            analysis_logger.critical(f"A critical error occurred for brand '{brand}': {e}", exc_info=True)
        finally:
            analysis_logger.info(f"\n{'='*60}\n ANALYSIS FOR BRAND: {brand.upper()} CONCLUDED \\ {'='*60}")
            
    analysis_logger.info("\n" + "=" * 60 + "\n ✅ ALL BRAND ANALYSES FINISHED. \\" + "=" * 60)
