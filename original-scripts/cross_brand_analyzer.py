import os
import google.generativeai as genai
from pathlib import Path

# --- Optional Libraries (with availability checks) ---
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


# --- CONFIGURATION ---
# Add all the report file paths to this list
REPORT_FILES = [
    "output_reports_v2/pagbank/pagbank_brand_report.html",
    "output_reports_v2/mcdonalds/mcdonalds_brand_report.html",
    "output_reports_v2/ifood/ifood_brand_report.html",
    "output_reports_v2/rexona/rexona_brand_report.html",
    "output_reports_v2/fiat/fiat_brand_report.html",
    "output_reports_v2/casas_bahia/casas_bahia_brand_report.html",
    "output_reports_v2/claro/claro_brand_report.html",
    "output_reports_v2/benegrip/benegrip_brand_report.html"
]

GEMINI_MODEL = "gemini-2.5-pro"
OUTPUT_FILENAME = "cross_brand_creative_analysis.html"

# --- HTML TEMPLATE ---
IDEAL_TEMPLATE = """<!DOCTYPE html><html lang="pt-br"><head><meta charset="UTF-8"><title>Análise Criativa Cross-Brand: Lições do Brasileirão 2025</title><style>body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; } .container { max-width: 1000px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); } h1, h2, h3 { color: #003366; border-bottom: 2px solid #ffc107; padding-bottom: 10px; margin-top: 30px; } h1 { text-align: center; font-size: 2.5em; border-bottom: none; } h2 { font-size: 1.8em; } h3 { font-size: 1.4em; border-bottom: none; } table { width: 100%; border-collapse: collapse; margin-top: 20px; } th, td { padding: 12px; border: 1px solid #ddd; text-align: left; } th { background-color: #f2f2f2; } img { max-width: 100%; height: auto; margin-top: 20px; border-radius: 8px; }</style></head><body><div class="container">{{ memo_html_content | safe }}</div></body></html>"""


# --- PROMPT ENGINEERING ---
ANALYSIS_PROMPT = """
You are a world-class Senior Marketing Strategist and your audience is an internal marketing team.

Your task is to perform a comprehensive, cross-brand analysis of creative performance during the Brasileirão 2025 sponsorship on CazéTV, based on the collection of individual brand reports provided below. Your analysis must be deep, insightful, and focused on delivering actionable creative principles for all our advertisers.

**Before you begin writing the report, take a deep breath and follow these steps:**
1.  First, methodically read through every individual brand report provided in the context below.
2.  As you read, internally synthesize the key findings, successes, and failures for each brand, paying close attention to the new data on Game Importance, Day of the Week, and Rational vs. Emotional message classifications.
3.  Only after you have processed all the reports, begin to construct the final markdown document based on the updated structure requested.

**Your final output must be a single, well-structured markdown document written in Brazilian Portuguese. Maintain a confident, authoritative, and data-driven tone throughout.**

Please structure your analysis to answer the following key questions, using the data and narratives from the reports as evidence.

---

# Análise Criativa Cross-Brand: Lições do Brasileirão 2025

## 1. A Grande Lição: Qual é a "Fórmula Vencedora" Universal?

Synthesize the findings from all brands to define a universal "winning formula." What is the single most important creative principle that advertisers should follow based on this data?

## 2. Análise de Contexto e Mensagens: O Que Realmente Conecta com a Audiência?

### O Impacto do Contexto do Jogo:
- Across all brands, what was the general impact of airing ads on **weekends vs. weekdays**?
- Did games with a higher **importance score** (e.g., classic matchups) consistently show better results for advertisers?

### O Duelo das Mensagens: Racional vs. Emocional
- Across all brands, which message type was more common: **Racional** (focada em ofertas, preços, funcionalidades) ou **Emocional** (focada em humor, identidade, momentos)?
- **Crucially, which approach drove better results?** Was there a clear winner in the battle for audience attention, or did the success depend on the brand's industry (e.g., rational for finance, emotional for CPG)?
- Provide examples from the reports to support your analysis.

### As Mensagens que Falharam:
- What types of messages consistently underperformed or generated negative lift across all brands?

## 3. Análise de Formatos de Anúncio: Onde a Mensagem Deve Viver?

### Os Formatos de Anúncio Mais Eficazes:
- Based on the reports, which ad formats delivered the best results across different brands (e.g., Anúncios de Vídeo, L-Shape, Gráficos Integrados)?
- Explain *why* these formats are effective in the context of a live football broadcast.

### Os Formatos que Não Funcionaram:
- Which formats were consistently associated with neutral or negative performance?
- What is the strategic implication? Should advertisers abandon these formats entirely?

## 4. Saúde da Marca a Longo Prazo: Estamos Construindo Valor Duradouro?
- Based on the trend analysis from the individual reports, is there a general pattern of **sustained, long-term growth** in baseline search interest for the sponsoring brands?
- Which brands, if any, are showing the strongest evidence of long-term brand building, and what can we learn from their strategy?

## 5. Recomendações Criativas Acionáveis para Todos os Anunciantes

Based on your entire analysis (including game context and long-term health), provide 3-4 sharp, actionable creative principles that the marketing team can share with every advertiser.
**Each principle must be a clear, direct instruction (e.g., "DO THIS..." or "AVOID THAT...") followed by a brief 'because...' justification referencing the analysis.** These should be universal rules for success.

---
**CONTEXT: INDIVIDUAL BRAND REPORTS**
"""

def setup_api_key():
    """Sets up the API key for Gemini."""
    try:
        api_key_path = Path('google_api_key.txt')
        api_key = os.environ.get('GOOGLE_API_KEY')
        if api_key_path.exists():
            api_key = api_key_path.read_text().strip()
        if not api_key:
            raise ValueError("API key not found in GOOGLE_API_KEY environment variable or google_api_key.txt")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Error setting up API key: {e}")
        return False

def render_html_report(memo_text, template_str, report_filename):
    """Renders the final HTML report from markdown text and a template."""
    print(f"Rendering final HTML report: {report_filename}...")
    if not all([JINJA_AVAILABLE, MARKDOWN_AVAILABLE]):
        print("Warning: Jinja2 or Markdown library not found. Saving raw text.")
        with open(report_filename.replace(".html", ".md"), "w", encoding='utf-8') as f:
            f.write(memo_text)
        return

    memo_html = markdown(memo_text, extensions=['tables'])
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    template = env.from_string(template_str)
    final_html = template.render(memo_html_content=memo_html)

    with open(report_filename, "w", encoding='utf-8') as f:
        f.write(final_html)
    print(f"✅ Report successfully generated: {report_filename}")


def run_cross_brand_analysis():
    """
    Reads all brand reports, combines them, sends them to Gemini for analysis,
    and saves the result as a structured HTML file.
    """
    if not setup_api_key():
        return

    print("Reading and compiling brand reports...")
    full_context = ""
    for report_path in REPORT_FILES:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                brand_name = Path(report_path).parent.name.upper()
                full_context += f"\n\n--- REPORT FOR {brand_name} ---\n\n{content}"
        except FileNotFoundError:
            print(f"Warning: Report file not found at {report_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading {report_path}: {e}")
            continue

    if not full_context:
        print("No report data could be loaded. Aborting analysis.")
        return

    final_prompt = ANALYSIS_PROMPT + full_context

    print("Sending data to Gemini for analysis. This may take a moment...")
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(final_prompt)

        render_html_report(response.text, IDEAL_TEMPLATE, OUTPUT_FILENAME)

    except Exception as e:
        print(f"An error occurred during the Gemini API call: {e}")

if __name__ == "__main__":
    run_cross_brand_analysis()