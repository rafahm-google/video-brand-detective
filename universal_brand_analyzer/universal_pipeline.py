# universal_pipeline.py
# Core engine for general brand detection in any video.

import os
import subprocess
import json
import csv
import time
import shutil
import logging
import datetime
import re
import multiprocessing
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import pandas as pd
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
class Config:
    """Centralized configuration for the universal brand analyzer."""
    STREAM_QUALITY: str = "360p"
    COOKIE_FILE_PATH: Optional[Path] = Path('cookies.txt')
    SAMPLE_RATE_SECONDS: float = 1.0  # Sample every second for general video
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    API_MAX_RETRIES: int = 5
    API_INITIAL_BACKOFF: int = 5
    CHUNK_SIZE_MINUTES: int = 2
    NUM_WORKERS: int = 8
    API_TIMEOUT_SECONDS: int = 600
    
    # Empty sets so the prompt defines what a brand is instead
    BRANDS_TO_EXCLUDE: Set[str] = set()
    FORMATS_TO_EXCLUDE: Set[str] = set()

    MAX_FRAME_DIMENSION: int = 512
    DEDUPLICATION_WINDOW_SECONDS: int = 15 # More robust window for generic detection
    MAIN_TEMP_DIR: Path = Path("universal_analysis_temp")

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.MAIN_TEMP_DIR, exist_ok=True)
        os.makedirs(cls.checkpoint_dir(), exist_ok=True)
        os.makedirs(cls.video_chunk_dir(), exist_ok=True)
        os.makedirs(cls.audio_chunk_dir(), exist_ok=True)

    @classmethod
    def video_chunk_dir(cls) -> Path: return cls.MAIN_TEMP_DIR / "video_chunks"
    @classmethod
    def audio_chunk_dir(cls) -> Path: return cls.MAIN_TEMP_DIR / "audio_chunks"
    @classmethod
    def checkpoint_dir(cls) -> Path: return cls.MAIN_TEMP_DIR / "chunk_results"

# --- LOGGING & API SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [UNIVERSAL] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_api_key() -> bool:
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            api_key_path = Path('google_api_key.txt')
            if api_key_path.exists(): api_key = api_key_path.read_text().strip()
        if not api_key: raise ValueError("API key not found")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Failed to configure API: {e}")
        return False

# --- CORE UTILS ---
def get_video_metadata(video_url: str) -> Optional[Tuple[str, str, int, datetime.datetime]]:
    logging.info(f"Fetching metadata for {video_url}...")
    try:
        if Path(video_url).exists():
            path = Path(video_url)
            return path.stem, path.stem, 0, datetime.datetime.now()

        command = ['yt-dlp', '-j', '--no-playlist', video_url]
        if Config.COOKIE_FILE_PATH and Config.COOKIE_FILE_PATH.exists(): command.extend(['--cookies', str(Config.COOKIE_FILE_PATH)])
        json_output = subprocess.check_output(command, text=True, encoding='utf-8')
        video_info = json.loads(json_output)
        video_id = video_info['id']
        video_title = video_info.get('title', f"video_{video_id}")
        duration = video_info.get('duration', 0)
        
        if ts := video_info.get('release_timestamp'):
            dt = datetime.datetime.fromtimestamp(ts)
        elif ds := video_info.get('upload_date'):
            dt = datetime.datetime.strptime(ds, '%Y%m%d')
        else:
            dt = datetime.datetime.now()
            
        return video_id, video_title, int(duration), dt
    except Exception as e:
        logging.error(f"Failed to get metadata: {e}")
        return None

def download_and_split(video_url: str, video_id: str) -> bool:
    local_video_path = Path(f"universal_video_{video_id}.mp4")
    local_audio_path = Path(f"universal_audio_{video_id}.mp3")
    
    try:
        if not local_video_path.exists():
            if Path(video_url).exists():
                shutil.copy(video_url, local_video_path)
            else:
                logging.info(f"Downloading video as '{local_video_path}'...")
                quality = Config.STREAM_QUALITY.replace("p", "")
                cmd = ['yt-dlp', '-f', f'bestvideo[height<={quality}]+bestaudio/best', '--merge-output-format', 'mp4', '-o', str(local_video_path), video_url]
                if Config.COOKIE_FILE_PATH and Config.COOKIE_FILE_PATH.exists():
                    cmd.extend(['--cookies', str(Config.COOKIE_FILE_PATH)])
                subprocess.run(cmd, check=True)
        
        if not local_audio_path.exists():
            logging.info(f"Extracting audio as '{local_audio_path}'...")
            cmd = ['ffmpeg', '-y', '-i', str(local_video_path), '-vn', '-acodec', 'libmp3lame', '-q:a', '2', str(local_audio_path), '-loglevel', 'error']
            subprocess.run(cmd, check=True)

        # Check if chunks already exist
        video_chunks = list(Config.video_chunk_dir().glob('*.mp4'))
        if len(video_chunks) > 0:
            logging.info(f"Found {len(video_chunks)} existing media chunks. Skipping segmentation.")
            return True

        logging.info("Chunking media...")
        segment_time = str(Config.CHUNK_SIZE_MINUTES * 60)
        cmd_video = ['ffmpeg', '-y', '-i', str(local_video_path), '-f', 'segment', '-segment_time', segment_time, '-reset_timestamps', '1', '-c', 'copy', str(Config.video_chunk_dir() / 'chunk_%03d.mp4'), '-loglevel', 'error']
        cmd_audio = ['ffmpeg', '-y', '-i', str(local_audio_path), '-f', 'segment', '-segment_time', segment_time, '-reset_timestamps', '1', '-c', 'copy', str(Config.audio_chunk_dir() / 'chunk_%03d.mp3'), '-loglevel', 'error']
        subprocess.run(cmd_video, check=True)
        subprocess.run(cmd_audio, check=True)
        return True
    except Exception as e:
        logging.error(f"Media prep failed: {e}")
        return False

# --- ANALYSIS ENGINE ---
def analyze_chunk(task_data: Tuple[int, Path, Path]) -> bool:
    chunk_index, video_path, audio_path = task_data
    if not setup_api_key(): return False
    
    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(fps * Config.SAMPLE_RATE_SECONDS))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((Config.MAX_FRAME_DIMENSION, Config.MAX_FRAME_DIMENSION))
                ts = (chunk_index * Config.CHUNK_SIZE_MINUTES * 60) + int(count / fps)
                frames.append({'image': img, 'seconds': ts})
            count += 1
        cap.release()
    except Exception as e:
        logging.error(f"Chunk {chunk_index} frame extraction failed: {e}")

    audio_file = None
    if audio_path.exists() and audio_path.stat().st_size > 100:
        try:
            audio_file = genai.upload_file(path=str(audio_path), display_name=f"univ_chunk_{chunk_index}")
        except Exception as e:
            logging.error(f"Chunk {chunk_index} audio upload failed: {e}")

    if not frames and not audio_file: return False

    chunk_start = chunk_index * Config.CHUNK_SIZE_MINUTES * 60
    prompt = f"""Você é um detetive de marcas generalista ultra-meticulo. 
SUA MISSÃO PRINCIPAL: Encontrar e catalogar **TODAS AS MARCAS** que aparecem ou são citadas neste vídeo.

INSTRUÇÕES OBRIGATÓRIAS:
1. **SEJA EXAUSTIVO**: Olhe em CADA lugar da tela. Procure por logos em camisetas, bonés, produtos no fundo da cena, outdoors, gráficos sobrepostos, textos na tela, carros, aparelhos e embalagens. Uma marca muito pequena ou rápida é tão crucial quanto uma grande.
2. **NÃO INVENTE**: Só liste marcas reais que estejam inequivocamente na tela ou no áudio. Use o nome correto da marca corporativa (ex: 'Nike', 'Apple', 'Coca-Cola').
3. **AUDIO**: Se a marca for citada, eu preciso saber exatamente o que foi dito. Transcreva a menção falada.
4. **TEXTO NA TELA**: Preste extrema atenção aos textos na tela referenciando marcas. Se houver texto descritivo escrito sobre ou próximo à marca (e.g., promoções, 'Apoio', hashtags), isso DEVE ser relatado integralmente.
5. **LOCALIZAÇÃO EXATA**: Para marcas visuais, você DEVE dizer EXATAMENTE onde a marca está na tela física ou gráfica. (ex: 'Canto inferior direito', 'No peito da camiseta preta do apresentador', 'Em um copo branco em cima da mesa', 'Como um overlay na tela inteira').
6. **TEMPO GLOBAL**: Os tempos em segundos indicados nos frames da sua imagem sāo absolutos. Para o audio, calcule seu tempo adicionando ao start_time ({chunk_start}s) deste pedaço.
7. **FORMATO JSON PURO**: Você DEVE retornar APENAS o objeto JSON e nenhuma outra frase. Se nada for encontrado, retorne as listas vazias: `{{"visual": [], "audio": []}}`.

RETORNO ESPERADO:
{{
    "visual": [
        {{
            "segundos_do_video": 123,
            "marca": "Nome da Marca Limpo",
            "formato": "Em que tipo de formato/suporte a marca estava (Ex: Logo na roupa, Overlay Gráfico, Embalagem)",
            "localizacao_exata": "Onde na tela exatamente (Ex: Canto superior direito colado à borda; No peito esquerdo da jaqueta)",
            "texto_na_tela_presente": "Transcreva ESPECIFICAMENTE os textos que aparecem próximos/sobre a logo. Se nenhum, 'N/A'",
            "detalhes": "Contexto geral do momento da detecção"
        }}
    ],
    "audio": [
        {{
            "segundos_do_video": 125,
            "marca": "Nome da Marca",
            "transcricao_exata": "Transcreva exatamente a frase ou parágrafo em que a marca foi mencionada",
            "contexto": "Quem falou e por quê"
        }}
    ]
}}
"""
    
    model = genai.GenerativeModel(Config.GEMINI_MODEL)
    payload = [prompt]
    for dict_img in frames:
        payload.extend([f"FRAME AT {dict_img['seconds']} SECONDS:", dict_img['image']])
    if audio_file: payload.append(audio_file)

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    visual_detections = []
    audio_detections = []

    for attempt in range(Config.API_MAX_RETRIES):
        wait_time = Config.API_INITIAL_BACKOFF * (2 ** attempt)
        try:
            res = model.generate_content(
                payload, 
                request_options={'timeout': Config.API_TIMEOUT_SECONDS},
                safety_settings=safety_settings
            )
            response_text = res.text.strip()
            
            data = None
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match: data = json.loads(match.group(0))
                else: raise

            for item in data.get('visual', []):
                visual_detections.append({
                    'Segundos Totais': int(item.get('segundos_do_video', 0)),
                    'Marca Identificada': item.get('marca', 'N/A'),
                    'Tipo de Detecção': 'Visual',
                    'Formato do Anúncio': item.get('formato', 'N/A'),
                    'Localização Exata na Tela': item.get('localizacao_exata', 'N/A'),
                    'Texto na Tela Presente': item.get('texto_na_tela_presente', 'N/A'),
                    'Resumo / Contexto': item.get('detalhes', ''),
                    'Transcrição da Menção (Audio)': 'N/A'
                })
            
            for item in data.get('audio', []):
                audio_detections.append({
                    'Segundos Totais': int(item.get('segundos_do_video', 0)),
                    'Marca Identificada': item.get('marca', 'N/A'),
                    'Tipo de Detecção': 'Áudio',
                    'Formato do Anúncio': 'Menção em Áudio',
                    'Localização Exata na Tela': 'N/A',
                    'Texto na Tela Presente': 'N/A',
                    'Resumo / Contexto': item.get('contexto', ''),
                    'Transcrição da Menção (Audio)': item.get('transcricao_exata', 'N/A')
                })
            
            break # Success! Escape retry loop
            
        except (google_exceptions.DeadlineExceeded, json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Chunk {chunk_index}, Attempt {attempt + 1}: Recoverable error ({type(e).__name__}). Retrying after {wait_time}s.")
        except Exception as e:
            logging.warning(f"Chunk {chunk_index}, Attempt {attempt + 1}: Unexpected error: {type(e).__name__}. Retrying after {wait_time}s.")
        
        if attempt + 1 == Config.API_MAX_RETRIES:
            logging.error(f"FATAL (Chunk {chunk_index}): Max retries reached. Skipping.")
            break
        time.sleep(wait_time)
            
    # --- Verification Layer (Option A with Batch Processing) ---
    if visual_detections:
        logging.info(f"Chunk {chunk_index}: Starting Batch Verification Layer for {len(visual_detections)} visual candidates...")
        
        # 1. Gather the union of unique frames to send
        unique_frames_to_send = {}
        for detect in visual_detections:
            target_sec = detect['Segundos Totais']
            target_seconds_list = [target_sec - 2, target_sec - 1, target_sec, target_sec + 1, target_sec + 2]
            
            for sec in target_seconds_list:
                closest_frame = min(frames, key=lambda x: abs(x['seconds'] - sec), default=None)
                if closest_frame:
                    unique_frames_to_send[closest_frame['seconds']] = closest_frame['image']
                    
        # Sort frames chronologically by timestamp
        sorted_frame_seconds = sorted(unique_frames_to_send.keys())
        verification_frames = [{'seconds': sec, 'image': unique_frames_to_send[sec]} for sec in sorted_frame_seconds]
        
        if not verification_frames:
            # Keep all candidates defensively if no frames found
            logging.warning(f"Chunk {chunk_index}: No verification frames could be extracted. Keeping candidates defensively.")
        else:
            # 2. Construct the single batch verification prompt listing all candidate claims
            claims_list_str = ""
            for idx, detect in enumerate(visual_detections):
                claims_list_str += f"Claim Index {idx}: Brand '{detect['Marca Identificada']}' at second {detect['Segundos Totais']} in format '{detect['Formato do Anúncio']}'\n"
                
            verify_prompt = f"""Você é um inspetor de qualidade ultra-rigoroso e analítico de detecção de marcas em vídeo.
Sua missão é validar uma lista de "Deteções Candidatas" feitas por um modelo primário. Você deve verificar cada deteção individualmente com base na sequência cronológica de imagens anexadas.

Lista de Deteções Candidatas a Verificar:
{claims_list_str}

Instruções Importantes de Validação:
1. Analise os frames próximos ao segundo de cada deteção candidata para confirmar se a marca reivindicada está de fato claramente visível.
2. Seja extremamente cético. Falsos positivos são inaceitáveis. Se você não tiver 100% de certeza absoluta de que a marca está claramente visível nos frames correspondentes àquela janela temporal, marque "verified": false.
3. Se a marca estiver presente, verifique se o formato de anúncio está correto. Se estiver errado ou for muito genérico, forneça o formato correto em "corrected_format".
4. Você DEVE responder APENAS com o JSON puro contendo a lista de resultados no formato estruturado abaixo. Sem introdução, sem blocos de código markdown (apenas JSON):

{{
    "verification_results": [
        {{
            "index": 0,  // O 'Claim Index' correspondente à deteção avaliada
            "verified": true,  // true se a marca de fato está lá com total certeza, false caso contrário
            "corrected_brand": "Nome da Marca",  // Manter o mesmo ou corrigir se aplicável
            "corrected_format": "Formato do Anúncio",  // Manter o mesmo ou corrigir se aplicável
            "confidence_explanation": "Uma breve justificativa técnica sobre sua decisão (Ex: 'Marca visível no peito da camisa no frame de X segundos')"
        }}
    ]
}}
"""
            verify_payload = [verify_prompt]
            for vf in verification_frames:
                verify_payload.extend([f"FRAME AT {vf['seconds']} SECONDS:", vf['image']])
                
            try:
                verify_res = model.generate_content(
                    verify_payload,
                    request_options={'timeout': Config.API_TIMEOUT_SECONDS},
                    safety_settings=safety_settings
                )
                verify_text = verify_res.text.strip()
                
                verify_data = None
                try:
                    verify_data = json.loads(verify_text)
                except json.JSONDecodeError:
                    match = re.search(r'\{.*\}', verify_text, re.DOTALL)
                    if match: verify_data = json.loads(match.group(0))
                    
                if verify_data and 'verification_results' in verify_data:
                    results_map = {item.get('index'): item for item in verify_data['verification_results'] if item.get('index') is not None}
                    
                    verified_visual_detections = []
                    for idx, detect in enumerate(visual_detections):
                        result = results_map.get(idx)
                        if result and result.get('verified') is True:
                            detect['Marca Identificada'] = result.get('corrected_brand', detect['Marca Identificada'])
                            detect['Formato do Anúncio'] = result.get('corrected_format', detect['Formato do Anúncio'])
                            detect['Resumo / Contexto'] = detect['Resumo / Contexto'] + f" [Verificado: {result.get('confidence_explanation', '')}]"
                            verified_visual_detections.append(detect)
                            logging.info(f"Chunk {chunk_index}: Candidate '{detect['Marca Identificada']}' VERIFIED at {detect['Segundos Totais']}s. Reason: {result.get('confidence_explanation')}")
                        else:
                            reason = result.get('confidence_explanation', 'Flagged as false positive') if result else 'No verification result returned'
                            logging.warning(f"Chunk {chunk_index}: Candidate '{detect['Marca Identificada']}' REJECTED at {detect['Segundos Totais']}s. Reason: {reason}")
                            
                    visual_detections = verified_visual_detections
                else:
                    logging.error(f"Chunk {chunk_index}: Bad JSON response or missing verification_results. Keeping candidates defensively.")
            except Exception as e:
                logging.error(f"Chunk {chunk_index}: Batch verification failed due to error: {e}. Keeping candidates defensively.")

    if audio_file:
        try: genai.delete_file(audio_file.name)
        except: pass
        
    checkpoint_path = Config.checkpoint_dir() / f"result_chunk_{chunk_index:03d}.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({"visual": visual_detections, "audio": audio_detections}, f, ensure_ascii=False, indent=2)
    
    return True

# --- REPORT GENERATION ---
def _count_unique_exposures(df: pd.DataFrame, window_seconds: int) -> int:
    if df.empty: return 0
    df = df.sort_values(by='Segundos Totais')
    count = 0
    last_timestamp = -window_seconds * 2
    for ts in df['Segundos Totais']:
        if (ts - last_timestamp) > window_seconds:
            count += 1
            last_timestamp = ts
    return count

def get_base_url(target: str, video_id: str) -> str:
    if Path(target).exists():
        return f"file://{Path(target).absolute()}"
    return f"https://www.youtube.com/watch?v={video_id}"

def build_reports(all_detections: List[Dict], video_id: str, title: str, dt: datetime.datetime, target: str):
    logging.info("Building deduplicated detailed and summary reports...")
    
    if not all_detections:
        logging.warning("No detections to build reports for.")
        return
        
    df = pd.DataFrame(all_detections)
    
    # Filter empty brands
    df = df.dropna(subset=['Segundos Totais', 'Marca Identificada'])
    df = df[df['Marca Identificada'].str.strip() != '']
    df['Marca Limpa'] = df['Marca Identificada'].str.strip().str.lower()
    
    # Exclude logic (currently empty, but functional if populated)
    df = df[~df['Marca Limpa'].isin(Config.BRANDS_TO_EXCLUDE)]
    
    df['Segundos Totais'] = pd.to_numeric(df['Segundos Totais'])
    df = df.sort_values(by='Segundos Totais').reset_index(drop=True)
    
    # --- Deduplication Logic ---
    logging.info(f"Deduplicating entries over {Config.DEDUPLICATION_WINDOW_SECONDS}s windows...")
    grouping_keys = ['Marca Limpa', 'Formato do Anúncio']
    df['time_diff'] = df.groupby(grouping_keys)['Segundos Totais'].diff()
    df_dedup = df[(df['time_diff'].isnull()) | (df['time_diff'] > Config.DEDUPLICATION_WINDOW_SECONDS)].copy()
    df_dedup.drop(columns=['time_diff'], inplace=True)
    
    df = df_dedup.sort_values(by='Segundos Totais').reset_index(drop=True)
    
    # --- File Output Prep ---
    output_dir = Path("universal_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    date_prefix = dt.strftime('%Y-%m-%d')
    base_url = get_base_url(target, video_id)
    
    # 1. Summary Report
    summary_path = output_dir / f"{date_prefix}_summary_{video_id}.csv"
    summary_rows = []
    
    for brand_name, brand_df in df.groupby('Marca Limpa'):
        display_name = brand_df['Marca Identificada'].mode()[0]
        vis_df = brand_df[brand_df['Tipo de Detecção'] == 'Visual']
        aud_df = brand_df[brand_df['Tipo de Detecção'] == 'Áudio']
        
        vis_count = _count_unique_exposures(vis_df.copy(), Config.DEDUPLICATION_WINDOW_SECONDS)
        aud_count = _count_unique_exposures(aud_df.copy(), Config.DEDUPLICATION_WINDOW_SECONDS)
        
        format_str = ", ".join([f"{count}x '{name}'" for name, count in Counter(brand_df['Formato do Anúncio']).items()])
        
        summary_rows.append([
            dt.strftime('%Y-%m-%d'), title, display_name,
            vis_count, aud_count, format_str
        ])
        
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Título", "Marca", "Frequência Visual", "Menções em Áudio", "Detalhamento de Formatos"])
        writer.writerows(sorted(summary_rows, key=lambda x: x[2]))
        
    # 2. Detailed Report
    detailed_path = output_dir / f"{date_prefix}_detailed_{video_id}.csv"
    detailed_data = []
    
    for _, row in df.iterrows():
        seconds = row['Segundos Totais']
        time_str = str(datetime.timedelta(seconds=int(seconds)))
        abs_ts = dt + datetime.timedelta(seconds=int(seconds))
        
        clip_link = f"{base_url}&t={int(seconds)}s" if "youtube.com" in base_url else base_url
        
        detailed_data.append({
            'Marca': row['Marca Identificada'],
            'Data Analise': dt.strftime('%Y-%m-%d'),
            'Tempo no Video': time_str,
            'Timestamp Absoluto': abs_ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Tipo Original': row['Tipo de Detecção'],
            'Formato': row['Formato do Anúncio'],
            'Localizacao Exata': row['Localização Exata na Tela'],
            'Texto na Tela Exibido': row['Texto na Tela Presente'],
            'Contexto Geral': row['Resumo / Contexto'],
            'O que foi Falado': row['Transcrição da Menção (Audio)'],
            'Link para o Momento': clip_link,
            'Titulo Video': title,
            'Source': base_url
        })
        
    df_detail = pd.DataFrame(detailed_data)
    df_detail.to_csv(detailed_path, index=False, quoting=csv.QUOTE_ALL)
    
    logging.info(f"Summary Report saved to {summary_path}")
    logging.info(f"Detailed Report saved to {detailed_path}")

# --- MAIN ORCHESTRATOR ---
def cleanup_temporary_files():
    logging.info("Cleaning up universal temporary files...")
    try:
        if Config.MAIN_TEMP_DIR.exists(): shutil.rmtree(Config.MAIN_TEMP_DIR)
    except Exception as e:
        logging.error(f"Temp cleanup failed: {e}")

def cleanup_cached_media(video_id: str):
    logging.info(f"Cleaning up heavy media bytes for {video_id}...")
    for p in Path('.').glob(f"universal_video_{video_id}.*"):
        try: p.unlink()
        except: pass
    for p in Path('.').glob(f"universal_audio_{video_id}.*"):
        try: p.unlink()
        except: pass

def run_universal_analysis(url: str):
    Config.setup_directories()
    metadata = get_video_metadata(url)
    if not metadata: 
        logging.error("Failed to gather metadata. Exiting.")
        return
        
    video_id, title, dur_secs, dt = metadata
    
    # Check if we already processed it
    date_prefix = dt.strftime('%Y-%m-%d')
    output_dir = Path("universal_reports")
    summary_path = output_dir / f"{date_prefix}_summary_{video_id}.csv"
    detailed_path = output_dir / f"{date_prefix}_detailed_{video_id}.csv"
    
    if summary_path.exists() and detailed_path.exists():
        logging.info(f"Files already exist for '{title}'. Skipping.")
        return

    # Phase 1: Prep media
    if not download_and_split(url, video_id): 
        logging.error("Failed to prepare media. Aborting analysis.")
        return
    
    # Phase 2: Distribute chunks
    video_chunks = sorted(Config.video_chunk_dir().glob('*.mp4'))
    audio_chunks = sorted(Config.audio_chunk_dir().glob('*.mp3'))
    
    completed = set()
    for f_path in Config.checkpoint_dir().glob('result_chunk_*.json'):
        match = re.search(r'result_chunk_(\d+)\.json', f_path.name)
        if match: completed.add(int(match.group(1)))
            
    num_total = min(len(video_chunks), len(audio_chunks))
    tasks = []
    for i in range(num_total):
        if i not in completed:
            tasks.append((i, video_chunks[i], audio_chunks[i]))
            
    if tasks:
        logging.info(f"Analyzing {len(tasks)} remaining chunks out of {num_total} total.")
        try:
            # Fork compatibility
            if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
                if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                     multiprocessing.set_start_method('spawn', force=True)
        except Exception: pass
        
        with multiprocessing.Pool(Config.NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(analyze_chunk, tasks), total=len(tasks), desc="Universal Analysis"))
    elif num_total > 0:
        logging.info("All chunks already complete in checkpoint folder.")

    # Phase 3: Reports and Cleanup
    all_results = []
    for f in sorted(Config.checkpoint_dir().glob('result_chunk_*.json')):
        with open(f, 'r', encoding='utf-8') as j:
            try:
                data = json.load(j)
                all_results.extend(data.get('visual', []))
                all_results.extend(data.get('audio', []))
            except json.JSONDecodeError:
                pass
                
    build_reports(all_results, video_id, title, dt, url)
    
    # Phase 4: Teardown
    cleanup_cached_media(video_id)
    cleanup_temporary_files()
    logging.info("Universal pipeline execution concluded successfully.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_universal_analysis(sys.argv[1])
