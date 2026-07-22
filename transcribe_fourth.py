from pathlib import Path
import json
from faster_whisper import WhisperModel

audio = Path('fourth_micro.ogg')
if not audio.exists() or audio.stat().st_size < 10000:
    raise RuntimeError(f'audio missing or too small: {audio.stat().st_size if audio.exists() else 0}')
model = WhisperModel('small', device='cpu', compute_type='int8')
segments, info = model.transcribe(str(audio), language='zh', beam_size=5, vad_filter=True, word_timestamps=True, initial_prompt='智慧体育体测分析与训练指导智能体，体测，BMI，坐位体前屈，Word，DOCX，知识库，训练建议，体育中考')
out=[]
for seg in segments:
    out.append({'start':round(seg.start,3),'end':round(seg.end,3),'text':seg.text.strip(),'words':[{'start':round(w.start,3) if w.start is not None else None,'end':round(w.end,3) if w.end is not None else None,'word':w.word} for w in (seg.words or [])]})
Path('transcript.json').write_text(json.dumps({'language':info.language,'duration':info.duration,'segments':out},ensure_ascii=False,indent=2),encoding='utf-8')
Path('transcript.txt').write_text('\n'.join(f"[{s['start']:.3f} --> {s['end']:.3f}] {s['text']}" for s in out),encoding='utf-8')
