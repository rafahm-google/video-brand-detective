# offline_analysis_pipeline.py
# This script is a modified version of the original analysis pipeline, designed to run
# in an environment without internet access to YouTube. It relies on pre-downloaded
# video files and a metadata CSV file.

import os
from dotenv import load_dotenv
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
import math

from google.api_core import exceptions as google_exceptions
import cv2
import pandas as pd
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURATION ---
load_dotenv()
class Config:
    """Centralized configuration for the analysis pipeline."""
    STREAM_QUALITY: str = "360p"
    # COOKIE_FILE_PATH is no longer needed for the offline pipeline
    SAMPLE_RATE_SECONDS: float = 0.5
    GEMINI_MODEL: str = "gemini-flash-latest"
    API_MAX_RETRIES: int = 5
    API_INITIAL_BACKOFF: int = 5
    CHUNK_SIZE_MINUTES: int = 3
    NUM_WORKERS: int = 10
    API_TIMEOUT_SECONDS: int = 600

    BRANDS_TO_EXCLUDE: Set[str] = {
        'cazé tv', 'cafétv', 'cafebv', 'cazétv', 'cazé', 'met life', 'youtube',
        'fifa', 'brasileirão', 'paulistão', 'campeonato brasileiro',
        'libertadores', 'copa libertadores da ameríca', 'campeonato paulista', 'cbf',
        'youtube premium', 'paulistão sicredi', 'neymar', 'neymar jr', 'guilherme', 
        'gabriel medina', 'bruninho', 'virgínia', 'zé felipe', 'santos', 'botafogo',
        'santos', 'corinthians', 'palmeiras', 'alexandre jesus'
    }
    FORMATS_TO_EXCLUDE: Set[str] = {'None'}
    MAX_FRAME_DIMENSION: int = 512
    DEDUPLICATION_WINDOW_SECONDS: int = 10
    MAIN_TEMP_DIR: Path = Path("analysis_temp")

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.MAIN_TEMP_DIR, exist_ok=True)
        os.environ['TMPDIR'] = str(cls.MAIN_TEMP_DIR)
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_api_key() -> bool:
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key: raise ValueError("API key not found in environment variables")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Failed to configure API: {e}")
        return False

# --- HELPER FUNCTIONS ---
def sanitize_filename(name: str) -> str: return re.sub(r'[<>:"/\\|?*]', '_', name)[:100].strip()

def prepare_media_from_local_file(local_video_path: Path, video_id: str, total_duration_seconds: int) -> bool:
    """
    Extracts audio and segments the local media file into chunks.
    It does not download anything.
    """
    if not local_video_path.exists():
        logging.error(f"Local video file not found: '{local_video_path}'. Cannot proceed.")
        return False

    local_audio_path = Path(f"audio_{video_id}.mp3")

    try:
        if not local_audio_path.exists():
            logging.info(f"Extracting full audio to '{local_audio_path}'...")
            cmd = ['ffmpeg', '-y', '-i', str(local_video_path), '-vn', '-acodec', 'libmp3lame', '-q:a', '2', str(local_audio_path), '-loglevel', 'error']
            subprocess.run(cmd, check=True)
        else:
            logging.info(f"Full audio '{local_audio_path}' already exists. Skipping extraction.")

        if total_duration_seconds > 0:
            expected_num_chunks = math.ceil(total_duration_seconds / (Config.CHUNK_SIZE_MINUTES * 60))
            existing_video_chunks = len(list(Config.video_chunk_dir().glob('*.mp4')))
            existing_audio_chunks = len(list(Config.audio_chunk_dir().glob('*.mp3')))

            if existing_video_chunks >= expected_num_chunks and existing_audio_chunks >= expected_num_chunks:
                logging.info(f"All {expected_num_chunks} media chunks already exist. Skipping segmentation.")
                return True
            else:
                logging.info("Chunking incomplete. Re-segmenting media...")
                for chunk_dir in [Config.video_chunk_dir(), Config.audio_chunk_dir()]:
                    if chunk_dir.exists(): shutil.rmtree(chunk_dir)
                    os.makedirs(chunk_dir)
        
        logging.info("Segmenting media into chunks...")
        segment_time = str(Config.CHUNK_SIZE_MINUTES * 60)
        cmd_video = ['ffmpeg', '-y', '-i', str(local_video_path), '-f', 'segment', '-segment_time', segment_time, '-reset_timestamps', '1', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'copy', str(Config.video_chunk_dir() / 'chunk_%03d.mp4'), '-loglevel', 'error']
        cmd_audio = ['ffmpeg', '-y', '-i', str(local_audio_path), '-f', 'segment', '-segment_time', segment_time, '-reset_timestamps', '1', '-c', 'copy', str(Config.audio_chunk_dir() / 'chunk_%03d.mp3'), '-loglevel', 'error']
        subprocess.run(cmd_video, check=True)
        subprocess.run(cmd_audio, check=True)
        return True

    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode(errors='ignore').strip() if e.stderr else "No stderr output."
        logging.error(f"A command failed during media preparation. Stderr: {error_output}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during media preparation: {e}")
        return False

# --- CORE ANALYSIS LOGIC (Identical to original pipeline) ---
def analyze_chunk(task_data: Tuple[int, Path, Path]) -> bool:
    chunk_index, video_chunk_path, audio_chunk_path = task_data
    if not setup_api_key(): return False
    
    frames_with_timestamps = []
    try:
        cap = cv2.VideoCapture(str(video_chunk_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_step = max(1, int(fps * Config.SAMPLE_RATE_SECONDS))
        frame_count = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret: break
            if frame_count % frame_step == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                pil_image.thumbnail((Config.MAX_FRAME_DIMENSION, Config.MAX_FRAME_DIMENSION))
                total_video_seconds = (chunk_index * Config.CHUNK_SIZE_MINUTES * 60) + int(frame_count / fps)
                frames_with_timestamps.append({'image': pil_image, 'seconds': total_video_seconds})
            frame_count += 1
        cap.release()
    except Exception as e:
        logging.error(f"Chunk {chunk_index}: Failed to extract frames: {e}")

    audio_file_for_api = None
    if audio_chunk_path.exists() and audio_chunk_path.stat().st_size > 100:
        try:
            audio_file_for_api = genai.upload_file(path=str(audio_chunk_path), display_name=f"chunk_{chunk_index}")
        except Exception as e:
            logging.error(f"Chunk {chunk_index}: Failed to upload audio: {e}")

    if not frames_with_timestamps and not audio_file_for_api: return False
    
    # --- CHANGE #1: Calculate the chunk's start time in seconds ---
    chunk_start_seconds = chunk_index * Config.CHUNK_SIZE_MINUTES * 60

    # --- CHANGE #2: Make the prompt an f-string and inject the start time into instruction #10 ---
    unified_prompt = f"""Você é um especialista em publicidade que analisa uma transmissão de futebol do Brasil.
SUA MISSÃO PRINCIPAL É SER METICULOSO. Não perca nenhum anúncio, mas sua prioridade máxima é a **PRECISÃO ABSOLUTA**, evitando falsos positivos.

INSTRUÇÕES OBRIGATÓRIAS:
1.  **REGRA DE OURO: NÃO INVENTE.** Se uma marca não estiver **CLARAMENTE e INEQUIVOCAMENTE** visível na imagem ou audível no áudio, **NÃO** a inclua na sua resposta. É melhor omitir uma detecção duvidosa do que reportar um anúncio que não existe.
2.  **LIDANDO COM ERROS:** Se você encontrar um segmento de vídeo particularmente confuso, com muitos cortes rápidos ou que não consiga analisar, em vez de gerar um erro, **retorne um JSON com listas vazias: `{{"visual": [], "audio": []}}`**. É preferível não reportar nada do que quebrar o fluxo da análise.
3.  **JSON PURO:** Sua resposta DEVE SER APENAS o objeto JSON, sem nenhum texto introdutório.
4.  **SEJA COMPLETO:** Identifique todas as marcas, mesmo as que aparecem brevemente ou como parte da interface da transmissão (placar, replays, etc.).
5.  **DISTINÇÃO CRÍTICA: Uma 'marca' é uma empresa, produto ou serviço (ex: PagBank, Clear, FIAT, McDonald's, Rexona, Jeep, Benegrip, Casas Bahia, Claro, Buscopan). Nomes de pessoas (jogadores, celebridades, etc., como Neymar, Virgínia) NÃO SÃO MARCAS e não devem ser incluídos na resposta. Além disso, se um anúncio mencionar uma linha de produtos junto com a marca (ex: 'Híbridos FIAT'), sua tarefa é extrair e reportar a marca principal ('FIAT').**
        * **McDonalds / Méqui:** O logo do McDonald's é o icônico "M" dourado, estilizado para se assemelhar a arcos, representando os "Arcos Dourados" da marca. Preste atenção a qualquer texto ou embalagem com "Méqui".
        * **iFood:** O logo do iFood apresenta o nome "iFood" em minúsculas brancas sobre um fundo ou elemento vermelho vibrante (frequentemente com um design de 'sorriso' embaixo), ou integrado em mensagens como "Pede iFood já".
        * **Rexona:** O logo da Rexona apresenta um proeminente símbolo de 'check' (visto) branco junto à palavra 'Rexona' em fonte branca, sobre um fundo retangular de cor verde-petróleo escuro, frequentemente acompanhado do slogan "Não te abandona".
        * **PagBank:** O logo do PagBank consiste em um ícone abstrato à esquerda e o nome "PagBank" em texto à direita; o ícone é composto por duas formas semicirculares sobrepostas que criam a ilusão de um "P" estilizado, com a forma de trás em tom de azul escuro e a da frente em um azul mais claro, quase ciano, e ambas contornadas por uma linha amarela.
        * **FIAT:** O logo da FIAT é composto pelo nome "FIAT" em letras maiúsculas, com um design moderno e limpo, onde o "F" tem uma barra horizontal estendida que atravessa o "I" e o "A", e a tipografia é robusta e contemporânea.
        * **Benegrip:** O logo da Benegrip apresenta o nome "BENEGRIP" em letras maiúsculas, com um design limpo e moderno, onde a palavra é escrita em verde vibrante sobre um fundo branco ou em um bloco verde claro.
        * **Casas Bahia:** O logo da Casas Bahia apresenta o nome "CASASBAHIA" em letras maiúsculas, com "CASAS" em azul e "BAHIA" em vermelho, e o nome é geralmente acompanhado por um slogan como "DEDICAÇÃO TOTAL A VOCÊ" e três ícones pequenos: um celular (app), um notebook (site) e uma loja (loja física).
        * **Claro:** O logo da Claro consiste em um círculo vermelho vibrante com a palavra "Claro" em branco dentro dele, seguida por um pequeno traço horizontal e um sinal de menos em tom de azul escuro.

6.  **IDENTIFIQUE O PATROCINADOR EM GRÁFICOS DIGITAIS:** Em QUALQUER gráfico digital sobreposto na transmissão (placar, replay, estatísticas, lower-thirds, pop-ups, banners, etc.), identifique a marca **patrocinadora**. Um logo ou nome de marca integrado *dentro* do gráfico (como parte do placar/timer, ou posicionado diretamente *ao lado*, *acima* ou *abaixo* dele como parte do mesmo overlay digital), é um 'Anúncio no Placar' se for adjacente ao placar ou timer. Se for um gráfico digital autônomo (não adjacente ao placar/timer), use a classificação mais apropriada (e.g., 'Pop-up / Overlay Gráfico na Tela', 'Banner no Rodapé').
7.  **ATENÇÃO REDOBRADA A FORMATOS ESPECIAIS E TEXTO NA TELA:**
    * Identifique anúncios em formato de 'L' que enquadram a tela (com ou sem QR code).
    * Identifique patrocínios de 'replay'.
    * Identifique banners de rodapé (lower-thirds).
    * Identifique patrocínio de segmentos como 'Garçom da Rodada'.
    * Identifique logos que aparecem **acima, ao lado ou abaixo de qualquer elemento da interface do jogo (placar, cronômetro, escalação, estatísticas, etc.) como parte da sobreposição digital**.
    * Identifique logos no calendário de próximos jogos ou em gráficos de estatísticas do jogo.
    * Identifique pop-ups e vídeos comerciais.
    * **MUITO IMPORTANTE: Preste atenção a qualquer texto ou slogan na tela que mencione uma marca, mesmo que o logo não seja o elemento principal do anúncio. Se o texto se refere a uma marca patrocinadora ou a uma campanha de marca, isso conta como uma inserção visual. Por exemplo, "LEÃO EM RECUPERAÇÃO! PagBank" ou "PEIXE LEVA VANTAGEM NO RETROSPECTO! PagBank".**

8.  **CLASSIFICAÇÃO VISUAL:** Para cada item em "visual", use EXATAMENTE uma destas categorias para `formato_do_anuncio`:
- **Quadro Patrocinado / Segmento de Conteúdo:** Anúncios que dão nome a um segmento específico e recorrente da transmissão, onde a marca é a patrocinadora oficial do quadro (Ex: 'Garçom da Rodada', 'Craque do Jogo', 'Homem em Campo').
- **Anúncio em Roupa de Torcedor:** Logotipo de patrocinador visível na camisa, boné, ou outra peça de roupa de um torcedor que a câmera foca.
- **Gráfico Integrado (Placar/Replay/Estatísticas/Agenda):** Logotipo integrado a elementos da interface da transmissão, como placares, replays, cronômetros, gráficos de estatísticas ou agendas de jogos futuros.
- **Banner/Overlay de Rodapé (Lower-third):** Faixas gráficas informativas ou promocionais que aparecem na parte inferior da tela.
- **Anúncio em Formato 'L' (L-Shape):** Gráficos que contornam a tela em forma de 'L', geralmente nas laterais e no rodapé. Anúncios com vídeo não estão neste grupo.
- **Anúncio em Vídeo:** Comerciais em formato de vídeo completo, normalmente exibidos durante pausas ou intervalos. Existem vídeos que não aparecem na tela toda, então fique atendo a comerciais de vídeo que apareçam parcialmente na tela, eles devem estar dentro deste grupo. Caso um anúncio em vídeo seja combinado com algum outro tipo de anúncio, como um banner, um anúncio em L ou outro tipo, você deve SEMPRE classificá-lo neste grupo. Ou seja, caso exista um vídeo no anúncio ele SEMPRE deverá estar neste grupo APENAS.
- **Anúncio Estático na Tela:** Logo e nome da marca destacados, geralmente sobre um fundo liso ou fixos em um canto da tela, sem estarem integrados a outro elemento gráfico.
- **Anúncio Call to Action (Genérico):** Anúncios que possuem uma chamada para ação clara e específica (ex: um QR code para um desconto) e que não se encaixam em outra categoria.
- **Integração em Comentário ao Vivo:** Anúncios posicionados dentro de segmentos de comentários ou análises ao vivo, fazendo a marca se tornar parte da conversa.
- **Anúncio no Campo:** Foco principal e intencional da câmera em patrocínios de uniforme, placas de LED, ou outros elementos físicos do campo.
- **Outros Tipos de Anúncio:** Use com extrema moderação apenas se NENHUMA outra categoria se aplicar.

9.  **SEGUNDOS PRECISOS:** O campo `segundos_do_video` deve ser um NÚMERO INTEIRO.
10. **PRECISÃO DE TEMPO PARA ÁUDIO:** Este trecho de áudio começa em **{chunk_start_seconds} segundos** do vídeo. Para qualquer detecção na lista "audio", o `segundos_do_video` deve ser o tempo global, calculado somando o tempo da menção *dentro deste trecho* ao tempo de início. **Exemplo: se a menção ocorre aos 48 segundos do áudio, o valor reportado deve ser {chunk_start_seconds + 48}.** Sua precisão é crítica.
11. **Capitalização da Marca:** A `marca_identificada` deve refletir a capitalização exata da marca conforme aparece visualmente ou é comumente utilizada (e.g., 'PagBank', 'FIAT', 'Clear').
12. **EXCLUSÃO DE ANÚNCIOS FÍSICOS:** Não inclua marcas que aparecem apenas como publicidade física no estádio (ex: em painéis LED no campo, placas estáticas) se não forem elementos digitais sobrepostos pela transmissão.
13. **MOMENTO DO ANÚNCIO:** Para cada anúncio, inclua o campo `momento_do_anuncio` com um dos seguintes valores:
    * `Pré-Jogo` - anúncios que apareceram antes da partida começar.
    * `Primeiro Tempo` - anúncios que aconteceram na primeira metade do jogo. Você pode tomar como base o tempo de jogo no placar e identificar se o anúncio aconteceu antes do fim do primeiro tempo, que normalmente vai do minuto 00:00 até 45:00, podendo ter acréscimos. Outra opção é avaliar se ao lado do tempo do jogo você encontra 1T.
    * `Intervalo` - anúncios que aconteceram após o primeiro tempo, mas antes do segundo tempo.
    * `Segundo Tempo` - anúncios que aconteceram na segunda metade do jogo. Você pode tomar como base o tempo de jogo no placar e identificar se o anúncio aconteceu antes do fim do segundo tempo, que normalmente vai do minuto 45:00 até 90:00, podendo ter acréscimos.Outra opção é avaliar se ao lado do tempo do jogo você encontra 2T.
    * `Pós-Jogo` - anúncios após o segundo tempo ou após o final do jogo.
14. **RESUMO DO ANUNCIO**: Escreva todos os pontos que foram mencionados no anuncio, quais os textos que estavam presente nas imagens e tambem a transcricao completa do audio. O seu objetivo aqui e monitorar todos os beneficios e pontos que foram listados pela marca em cada anuncio, se existiam ofertas, precos diferentes, quais produtos foram citados, quais marcas apareceram e etc.
15. **VALIDAÇÃO FINAL OBRIGATÓRIA (CHECKLIST):** Antes de finalizar, verifique sua resposta com base neste checklist. Se você esqueceu algo, corrija o JSON antes de terminar:
    * **CHECK 1: Banners de Rodapé:** Verifiquei cuidadosamente a parte inferior da tela em todos os frames em busca de banners, como os que mostram estatísticas do jogo patrocinadas (ex: 'Leão em Recuperação!')?
    * **CHECK 2: Gráficos de Placar/Tempo:** Verifiquei se há algum logo de patrocinador permanentemente ao lado, acima ou abaixo do placar e do cronômetro?
    * **CHECK 3: Anúncios em 'L':** Verifiquei as bordas da tela em busca de anúncios em formato 'L'?
    * **CHECK 4: Anúncios de Pop-up:** Verifiquei se houve alguma aparição de um logo ao redor da tela durante os frames analisados?
    * **CHECK 5: Anúncios em Vídeo:** Verifiquei se houve algum anúncio em vídeo durante os frames analisados?
    * **CHECK 6: Precisão do JSON:** Meu JSON final está perfeitamente formatado, sem vírgulas sobrando no final das listas?
Formato JSON esperado com exemplos variados e abrangentes:
```json
{{
    "visual": [
        {{"segundos_do_video": 3813, "marca_identificada": "Claro", "formato_do_anuncio": "Gráfico Integrado (Placar/Replay/Estatísticas/Agenda)", "resumo_do_anuncio": "Logo da Claro exibido em um box vermelho, posicionado diretamente ao lado do placar do jogo.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}},
        {{"segundos_do_video": 2745, "marca_identificada": "PagBank", "formato_do_anuncio": "Gráfico Integrado (Placar/Replay/Estatísticas/Agenda)", "resumo_do_anuncio": "Logo da PagBank exibido em um banner amarelo posicionado diretamente acima do placar do jogo.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Primeiro Tempo"}},
        {{"segundos_do_video": 2350, "marca_identificada": "Clear", "formato_do_anuncio": "Anúncio em Formato 'L' (L-Shape)", "resumo_do_anuncio": "Anúncio em formato 'L' na lateral direita e rodapé da tela, com o jogador Vini Jr. e o produto Clear Men.", "chamada_para_acao": "Os melhores usam o Nº1. 3x Poder Anticaspa.", "momento_do_anuncio": "Primeiro Tempo"}},
        {{"segundos_do_video": 903, "marca_identificada": "FIAT", "formato_do_anuncio": "Gráfico Integrado (Placar/Replay/Estatísticas/Agenda)", "resumo_do_anuncio": "Logo 'Híbridos FIAT' exibido como parte integrante do overlay do placar do jogo, no centro da tela.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Pré-Jogo"}},
        {{"segundos_do_video": 3555, "marca_identificada": "Claro", "formato_do_anuncio": "Gráfico Integrado (Placar/Replay/Estatísticas/Agenda)", "resumo_do_anuncio": "Logo da Claro exibido em um overlay vermelho durante o 'REPLAY' de uma jogada.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}},
        {{"segundos_do_video": 1236, "marca_identificada": "PagBank", "formato_do_anuncio": "Banner/Overlay de Rodapé (Lower-third)", "resumo_do_anuncio": "Banner de rodapé (lower-third) com o logo da PagBank, exibindo o texto 'HISTÓRICO FAVORÁVEL AO PEIXE!'.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Primeiro Tempo"}},
        {{"segundos_do_video": 685, "marca_identificada": "Clear", "formato_do_anuncio": "Anúncio em Vídeo", "resumo_do_anuncio": "Comercial da Clear Men com Vini Jr. exibido em uma grande janela na tela, ao lado da agenda de jogos da CazéTV.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Intervalo"}},
        {{"segundos_do_video": 2464, "marca_identificada": "FIAT", "formato_do_anuncio": "Banner/Overlay de Rodapé (Lower-third)", "resumo_do_anuncio": "Banner de rodapé azul (lower-third) com o texto 'Híbridos FIAT' e a informação 'HISTÓRICO FAVORÁVEL AO PEIXE!'.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Primeiro Tempo"}},
        {{"segundos_do_video": 8888, "marca_identificada": "PagBank", "formato_do_anuncio": "Quadro Patrocinado / Segmento de Conteúdo", "resumo_do_anuncio": "Gráfico de tela cheia para o segmento 'Garçom da Rodada', oferecido pela PagBank.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Pós-Jogo"}},
        {{"segundos_do_video": 659, "marca_identificada": "Casas Bahia", "formato_do_anuncio": "Anúncio em Formato 'L' (L-Shape)", "resumo_do_anuncio": "Anúncio em formato 'L' na lateral e rodapé da tela com QR code, promovendo 'Dia das Mães'.", "chamada_para_acao": "Aqui tem o presente ideal para a sua mãe a partir de R$89,00.", "momento_do_anuncio": "Pré-Jogo"}},
        {{"segundos_do_video": 8286, "marca_identificada": "FIAT", "formato_do_anuncio": "Gráfico Integrado (Placar/Replay/Estatísticas/Agenda)", "resumo_do_anuncio": "Logo da FIAT exibido em um pequeno overlay diretamente abaixo do placar do jogo.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}},
        {{"segundos_do_video": 5766, "marca_identificada": "PagBank", "formato_do_anuncio": "Banner/Overlay de Rodapé (Lower-third)", "resumo_do_anuncio": "Logo da PagBank ao lado do texto 'LEÃO EM RECUPERAÇÃO!' e 'FORTALEZA VEM DE TRÊS DERROTAS CONSECUTIVAS NO BRASILEIRÃO!' no rodapé da tela.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}},
        {{"segundos_do_video": 5585, "marca_identificada": "PagBank", "formato_do_anuncio": "Banner/Overlay de Rodapé (Lower-third)", "resumo_do_anuncio": "Logo da PagBank ao lado do texto 'LEÃO EM RECUPERAÇÃO!' e 'FORTALEZA TEM 43% DE APROVEITAMENTO NA TEMPORADA DE 2025!' no rodapÃ© da tela.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}},
        {{"segundos_do_video": 5594, "marca_identificada": "PagBank", "formato_do_anuncio": "Banner/Overlay de Rodapé (Lower-third)", "resumo_do_anuncio": "Logo da PagBank ao lado do texto 'LEÃO EM RECUPERAÇÃO!' e 'FORTALEZA TEM 43% DE APROVEITAMENTO NA TEMPORADA DE 2025!' no rodapÃ© da tela.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}},
        {{"segundos_do_video": 5590, "marca_identificada": "PagBank", "formato_do_anuncio": "Banner/Overlay de Rodapé (Lower-third)", "resumo_do_anuncio": "Logo da PagBank ao lado do texto 'PEIXE LEVA VANTAGEM NO RETROSPECTO!' e informações de jogos no rodapÃ© da tela.", "chamada_para_acao": "N/A", "momento_do_anuncio": "Segundo Tempo"}}
    ],
    "audio": [
        {{"segundos_do_video": 456, "marca_identificada": "Sportingbet", "transcricao_da_mencao": "Faz um Sportingbet aí!", "contexto_da_mencao": "Narrador incentiva apostas.", "momento_do_anuncio": "Primeiro Tempo"}}
    ]
}}
"""

    model = genai.GenerativeModel(Config.GEMINI_MODEL)
    api_payload = [unified_prompt]
    if frames_with_timestamps:
        for item in frames_with_timestamps:
            api_payload.extend([f"FRAME AT {item['seconds']} SECONDS:", item['image']])
    if audio_file_for_api: api_payload.append(audio_file_for_api)

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    visual_detections, audio_detections = [], []
    for attempt in range(Config.API_MAX_RETRIES):
        wait_time = Config.API_INITIAL_BACKOFF * (2 ** attempt)
        try:
            response = model.generate_content(
                api_payload,
                request_options={'timeout': Config.API_TIMEOUT_SECONDS},
                safety_settings=safety_settings
            )
            response_text = response.text.strip()
            
            data = None
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match: data = json.loads(match.group(0))
                else: raise
            
            for item in data.get('visual', []): visual_detections.append({'Segundos Totais': int(item.get('segundos_do_video', 0)), 'Horário no Vídeo': str(datetime.timedelta(seconds=int(item.get('segundos_do_video', 0)))), 'Marca Identificada': item.get('marca_identificada'), 'Formato do Anúncio': item.get('formato_do_anuncio'), 'Resumo do Anúncio': item.get('resumo_do_anuncio'), 'Chamada para Ação': item.get('chamada_para_acao'), 'Momento do Anúncio': item.get('momento_do_anuncio', 'N/A')})
            for item in data.get('audio', []): audio_detections.append({'Segundos Totais': int(item.get('segundos_do_video', 0)), 'Horário no Vídeo': str(datetime.timedelta(seconds=int(item.get('segundos_do_video', 0)))), 'Marca Identificada': item.get('marca_identificada'), 'Formato do Anúncio': 'Menção em Áudio', 'Resumo do Anúncio': f"Contexto: {item.get('contexto_da_mencao')}", 'Chamada para Ação': f"Transcrição: {item.get('transcricao_da_mencao')}", 'Momento do Anúncio': item.get('momento_do_anuncio', 'N/A')})
            break
        except (google_exceptions.DeadlineExceeded, json.JSONDecodeError, ValueError) as e:
             logging.warning(f"Chunk {chunk_index}, Attempt {attempt + 1}: Recoverable error ({type(e).__name__}). Retrying after {wait_time}s.")
        except Exception as e:
            logging.warning(f"Chunk {chunk_index}, Attempt {attempt + 1}: An unexpected error occurred: {type(e).__name__}. Retrying after {wait_time}s.")
        
        if attempt + 1 == Config.API_MAX_RETRIES:
            logging.error(f"FATAL (Chunk {chunk_index}): Max retries reached. Skipping chunk.")
            break
        time.sleep(wait_time)
            
    if audio_file_for_api:
        try: genai.delete_file(audio_file_for_api.name)
        except Exception: pass
        
    checkpoint_path = Config.checkpoint_dir() / f"result_chunk_{chunk_index:03d}.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f: json.dump({"visual": visual_detections, "audio": audio_detections}, f, ensure_ascii=False, indent=2)
    return True


# --- REPORTING FUNCTIONS ---
def _count_unique_exposures(df: pd.DataFrame, window_seconds: int) -> int:
    if df.empty: return 0
    df['TimeDelta'] = pd.to_timedelta(df['Horário no Vídeo'], errors='coerce').dropna()
    df = df.sort_values(by='TimeDelta')
    count = 0
    last_timestamp = pd.Timedelta(seconds=-window_seconds * 2)
    for ts in df['TimeDelta']:
        if (ts - last_timestamp).total_seconds() > window_seconds: count += 1; last_timestamp = ts
    return count

def create_summary_report(df: pd.DataFrame, output_path: Path, video_title: str, video_upload_datetime: datetime.datetime) -> None:
    logging.info("Generating summary report...")
    if df.empty:
        logging.warning("DataFrame is empty, skipping summary report generation.")
        return
        
    header = [
        "Data do Jogo", "Título do Vídeo", "Marca", 
        "Frequência Visual (Exposições Únicas)", 
        "Menções em Áudio (Exposições Únicas)", 
        "Detalhamento dos Formatos"
    ]
    report_rows = []
    date_str = video_upload_datetime.strftime('%Y-%m-%d')

    for brand_name, brand_df in df.groupby('Marca Limpa'):
        visual_df = brand_df[brand_df['Formato do Anúncio'] != 'Menção em Áudio']
        audio_df = brand_df[brand_df['Formato do Anúncio'] == 'Menção em Áudio']
        
        unique_visual_count = _count_unique_exposures(visual_df.copy(), Config.DEDUPLICATION_WINDOW_SECONDS)
        unique_audio_count = _count_unique_exposures(audio_df.copy(), Config.DEDUPLICATION_WINDOW_SECONDS)
        
        if unique_visual_count == 0 and unique_audio_count == 0:
            continue
            
        format_details = ", ".join([f"{count}x '{name}'" for name, count in Counter(brand_df['Formato do Anúncio']).items()])
        
        report_rows.append([
            date_str, video_title, brand_name.title(), 
            unique_visual_count, unique_audio_count, format_details
        ])
        
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sorted(report_rows, key=lambda x: x[2])) # Sort by brand name
        
    logging.info(f"Summary report saved to '{output_path}'.")

def create_detailed_report(all_detections: List[Dict], video_id: str, video_title: str, video_upload_datetime: datetime.datetime, output_path: Path) -> None:
    logging.info("Generating unified detailed report...")
    if not all_detections: return
    report_data = []
    base_video_url = f"https://www.youtube.com/watch?v={video_id}"
    date_str = video_upload_datetime.strftime('%Y-%m-%d')
    for ad in all_detections:
        seconds = int(ad.get('Segundos Totais', 0))
        timestamp = video_upload_datetime + datetime.timedelta(seconds=seconds)
        is_audio = ad.get('Formato do Anúncio') == 'Menção em Áudio'
        report_data.append({
            'sponsor': ad.get('Marca Identificada'), 
            'Date': date_str, 
            'Horário no Vídeo': ad.get('Horário no Vídeo'), 
            'Timestamp Absoluto': timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
            'Momento no Jogo': ad.get('Momento do Anúncio', 'N/A'),
            'Tipo de Detecção': 'Áudio' if is_audio else 'Visual', 
            'Formato / Contexto': ad.get('Formato do Anúncio', 'N/A'),
            'Resumo do Anúncio': ad.get('Resumo do Anúncio', 'N/A'),
            'Detalhe / CTA / Transcrição': ad.get('Chamada para Ação', 'N/A'), 
            'Link para o Momento': f"{base_video_url}&t={seconds}s", 
            'Video Title': video_title, 
            'Video URL': base_video_url
        })
    df = pd.DataFrame(report_data)
    # Reorder columns to place 'Momento no Jogo' in a logical spot
    column_order = [
        'sponsor', 'Date', 'Horário no Vídeo', 'Timestamp Absoluto', 'Momento no Jogo',
        'Tipo de Detecção', 'Formato / Contexto', 'Resumo do Anúncio', 'Detalhe / CTA / Transcrição',
        'Link para o Momento', 'Video Title', 'Video URL'
    ]
    df = df[column_order]
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    logging.info(f"Unified detailed report saved to '{output_path}'.")

# --- CLEANUP FUNCTIONS (Identical to original pipeline) ---
def cleanup_temporary_files():
    """Removes the main temporary directory to ensure a clean slate."""
    try:
        if Config.MAIN_TEMP_DIR.exists():
            shutil.rmtree(Config.MAIN_TEMP_DIR)
            logging.info(f"Successfully cleaned up temporary directory: {Config.MAIN_TEMP_DIR}")
    except Exception as e:
        logging.warning(f"Could not clean up temporary directory {Config.MAIN_TEMP_DIR}: {e}")
def cleanup_cached_media(video_id: str):
    # ... (identical to original)
    pass
def force_clear_specific_files_from_trash(video_id: str):
    # ... (identical to original)
    pass

# --- OFFLINE ANALYSIS ENGINE ---
def _run_offline_analysis_engine(local_video_path: Path, metadata: Dict) -> bool:
    video_id = metadata['video_id']
    video_title = metadata['title']
    duration_seconds = int(metadata['duration_seconds'])
    video_upload_datetime = datetime.datetime.fromisoformat(metadata['upload_datetime'])

    if not prepare_media_from_local_file(local_video_path, video_id, duration_seconds):
        logging.error("Failed to prepare media from local file. Aborting analysis.")
        return False
    
    video_chunks = sorted(Config.video_chunk_dir().glob('*.mp4'))
    audio_chunks = sorted(Config.audio_chunk_dir().glob('*.mp3'))
    
    completed_chunks = set()
    for f_path in Config.checkpoint_dir().glob('result_chunk_*.json'):
        match = re.search(r'result_chunk_(\d+)\.json', f_path.name)
        if match: completed_chunks.add(int(match.group(1)))

    tasks_to_run = [(i, video_chunks[i], audio_chunks[i]) for i in range(min(len(video_chunks), len(audio_chunks))) if i not in completed_chunks]

    if not tasks_to_run and len(completed_chunks) > 0:
        logging.info("All chunks have been processed previously. Moving to report generation.")
    elif tasks_to_run:
        logging.info(f"Resuming analysis for {len(tasks_to_run)} remaining chunks.")
        with multiprocessing.Pool(processes=Config.NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(analyze_chunk, tasks_to_run), total=len(tasks_to_run), desc="Analyzing Media Chunks"))
    
    all_detections = []
    for f_path in sorted(Config.checkpoint_dir().glob('*.json')):
        try:
            with open(f_path, 'r', encoding='utf-8') as f: data = json.load(f)
            all_detections.extend(data.get("visual", [])); all_detections.extend(data.get("audio", []))
        except (json.JSONDecodeError, FileNotFoundError): 
            logging.warning(f"Corrupt or missing checkpoint ignored: {f_path}")

    if not all_detections:
        logging.warning("No detections found after processing. No reports will be generated.")
        return True

    df = pd.DataFrame(all_detections)
    df = df.dropna(subset=['Segundos Totais', 'Marca Identificada'])
    df = df[df['Marca Identificada'].str.strip() != '']
    df['Marca Limpa'] = df['Marca Identificada'].str.strip().str.lower()
    df = df[~df['Marca Limpa'].isin(Config.BRANDS_TO_EXCLUDE)]
    df = df[~df['Formato do Anúncio'].isin(Config.FORMATS_TO_EXCLUDE)]
    
    if df.empty:
        logging.warning("No relevant detections remain after initial filtering.")
        return True

    df['Segundos Totais'] = pd.to_numeric(df['Segundos Totais'])
    df = df.sort_values(by='Segundos Totais').reset_index(drop=True)

    logging.info(f"Deduplicating raw results ({len(df)} rows) using a {Config.DEDUPLICATION_WINDOW_SECONDS}s window...")
    grouping_keys = ['Marca Limpa', 'Formato do Anúncio']
    df['time_diff'] = df.groupby(grouping_keys)['Segundos Totais'].diff()
    df_deduplicated = df[(df['time_diff'].isnull()) | (df['time_diff'] > Config.DEDUPLICATION_WINDOW_SECONDS)].copy()
    df_deduplicated.drop(columns=['time_diff'], inplace=True)
    logging.info(f"Deduplication complete. Reduced to {len(df_deduplicated)} unique ad appearances.")
    
    if df_deduplicated.empty:
        logging.warning("No relevant detections remain after deduplication.")
        return True
    
    df = df_deduplicated.sort_values(by='Segundos Totais').reset_index(drop=True)

    output_dir = Path("~/DriveFileStream/My Drive/Brasileirao Files/output_reports_v3").expanduser()
    os.makedirs(output_dir, exist_ok=True)
    
    date_prefix = video_upload_datetime.strftime('%Y-%m-%d')
    summary_report_path = output_dir / f"{date_prefix}_summary_report_{video_id}.csv"
    detailed_report_path = output_dir / f"{date_prefix}_detailed_log_{video_id}.csv"

    create_summary_report(df.copy(), summary_report_path, video_title, video_upload_datetime)
    create_detailed_report(df.to_dict('records'), video_id, video_title, video_upload_datetime, detailed_report_path)
    
    return True

# --- MAIN OFFLINE EXECUTION BLOCK ---
def main_offline(video_metadata: Dict) -> Tuple[bool, Optional[str]]:
    """Main function to run the full OFFLINE analysis for a single video."""
    video_id = video_metadata.get('video_id')
    
    # --- CRITICAL FIX: Ensure clean state for new video ---
    # We clean up any potential leftovers from a previous video (e.g. if the runner moved on after a failure)
    # This prioritizes correctness (no cross-contamination) over resuming a different video's failed state.
    cleanup_temporary_files()
    
    local_path = Path(video_metadata.get('local_path'))
    
    Config.setup_directories() 

    if not setup_api_key():
        logging.error("API Key setup failed. Cannot proceed.")
        return False, None
    
    if not local_path.exists():
        logging.error(f"Local video file not found at '{local_path}'. Cannot proceed.")
        return False, video_id

    output_dir = Path("~/DriveFileStream/My Drive/Brasileirao Files/output_reports_v3").expanduser()
    date_prefix = datetime.datetime.fromisoformat(video_metadata['upload_datetime']).strftime('%Y-%m-%d')
    summary_report_path = output_dir / f"{date_prefix}_summary_report_{video_id}.csv"
    detailed_report_path = output_dir / f"{date_prefix}_detailed_log_{video_id}.csv"

    if summary_report_path.exists() and detailed_report_path.exists():
        logging.info(f"Final reports for video ID '{video_id}' already exist. SKIPPING.")
        return True, video_id

    try:
        if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
             multiprocessing.set_start_method('spawn', force=True)
        else:
             multiprocessing.set_start_method('fork', force=True)
    except (ValueError, AttributeError, RuntimeError):
        pass

    success = False
    try:
        logging.info(f"Starting OFFLINE analysis for video ID '{video_id}'.")
        success = _run_offline_analysis_engine(local_path, video_metadata)
    except Exception as e:
        logging.critical(f"An unhandled error occurred during the offline analysis of {video_id}. Error: {e}", exc_info=True)
    finally:
        logging.info(f"Finished processing for video_id: {video_id}. Final Status: {'Success' if success else 'Failed'}")

    if success:
        logging.info("Analysis completed. Cleaning up temporary and cached files.")
        cleanup_cached_media(video_id)
        cleanup_temporary_files()
    else:
        logging.warning("Analysis failed. Temporary files are being kept for resume.")
        
    return success, video_id
