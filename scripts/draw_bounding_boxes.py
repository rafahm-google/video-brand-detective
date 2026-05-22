import sys
import os
import cv2
import pandas as pd
import json
import logging
import subprocess
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def parse_bbox(bbox_str):
    if pd.isna(bbox_str):
        return None
    try:
        return json.loads(bbox_str.replace("'", '"'))
    except:
        try:
            match = re.search(r'\[([\d\s,]+)\]', bbox_str)
            if match:
                return [int(x.strip()) for x in match.group(1).split(',')]
            return None
        except:
            return None

def main():
    if len(sys.argv) < 3:
        print("Uso: python draw_bounding_boxes.py <caminho_do_video> <caminho_do_csv>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    if not video_path.exists():
        logging.error(f"Vídeo não encontrado: {video_path}")
        sys.exit(1)
        
    if not csv_path.exists():
        logging.error(f"CSV não encontrado: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    
    def time_str_to_seconds(t_str):
        if pd.isna(t_str):
            return -1
        try:
            if ':' in t_str:
                parts = t_str.split(':')
                return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
            else:
                return int(t_str)
        except:
            return -1

    if 'Tempo no Video' in df.columns:
        df['segundos_real'] = df['Tempo no Video'].apply(time_str_to_seconds)
    else:
        logging.error("Coluna 'Tempo no Video' não encontrada no CSV gerado.")
        sys.exit(1)
        
    detections_by_second = {}
    
    for _, row in df.iterrows():
        tipo = row.get('Tipo Original', '')
        if tipo != 'Visual':
            continue
            
        sec = row['segundos_real']
        if sec < 0:
            continue
            
        bbox_str = str(row.get('Localizacao Exata', ''))
        bbox = parse_bbox(bbox_str)
        
        if bbox and len(bbox) == 4:
            if sec not in detections_by_second:
                detections_by_second[sec] = []
            detections_by_second[sec].append({
                'marca': str(row.get('Marca', 'Desconhecida')),
                'bbox': bbox
            })
            
    logging.info(f"Encontradas {sum(len(v) for v in detections_by_second.values())} caixas visuais extraíveis do CSV.")

    temp_h264 = None
    cap = cv2.VideoCapture(str(video_path))
    ret, _ = cap.read()
    if not ret:
        logging.warning("OpenCV falhou ao ler o vídeo original (possível codec AV1). Transcodificando para H.264 temporariamente...")
        cap.release()
        temp_h264 = f"temp_h264_{video_path.name}"
        subprocess.run(['ffmpeg', '-y', '-i', str(video_path), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', temp_h264, '-loglevel', 'error'], check=True)
        cap = cv2.VideoCapture(temp_h264)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = f"annotated_temp_{video_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    logging.info(f"Iniciando desenho de box no vídeo ({width}x{height}) - FPS: {fps:.2f}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_second = int(frame_count / fps)
        
        if current_second in detections_by_second:
            for d in detections_by_second[current_second]:
                ymin_n, xmin_n, ymax_n, xmax_n = d['bbox']
                # Ajusta caso o modelo retorne em ordem inversa de escala, ou com valores fora
                ymin_n, xmin_n, ymax_n, xmax_n = max(0, ymin_n), max(0, xmin_n), min(1000, ymax_n), min(1000, xmax_n)
                
                xmin = int((xmin_n / 1000.0) * width)
                ymin = int((ymin_n / 1000.0) * height)
                xmax = int((xmax_n / 1000.0) * width)
                ymax = int((ymax_n / 1000.0) * height)
                
                marca = d['marca']
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4) # Red color
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(marca, font, font_scale, thickness)
                
                title_bg_target_y = max(0, ymin - text_height - 10)
                if ymin < 40: # Se for muito no topo, desenha abaixo
                    title_bg_target_y = ymax
                    ymin_text = ymax + text_height + 5
                else:
                    ymin_text = ymin - 5
                    
                cv2.rectangle(frame, (xmin, title_bg_target_y), (xmin + text_width + 10, title_bg_target_y + text_height + 10), (0, 0, 255), -1)
                cv2.putText(frame, marca, (xmin + 5, ymin_text), font, font_scale, (255, 255, 255), thickness)

        out.write(frame)
        frame_count += 1
        if frame_count > 0 and frame_count % int(fps * 30) == 0:
             logging.info(f"Processados {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    logging.info("Processamento de quadros do vídeo concluído. Mesclando áudio...")

    final_output = video_path.parent / f"{video_path.stem}_annotated.mp4"
    cmd = [
        'ffmpeg', '-y',
        '-i', output_video_path,
        '-i', str(video_path),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?', 
        str(final_output),
        '-loglevel', 'error'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Vídeo final com áudio e anotações foi gerado com sucesso: {final_output}")
        os.remove(output_video_path)
    except Exception as e:
        logging.error(f"Erro ao juntar áudio com ffmpeg: {e}")
        logging.info(f"O vídeo temporário sem áudio foi mantido: {output_video_path}")

    if temp_h264 and os.path.exists(temp_h264):
        os.remove(temp_h264)

if __name__ == "__main__":
    main()
