"""LocateAnything-3B smoke test on real-robot camera_top frames (LeRobot v3)."""
import json, re, sys, time
from pathlib import Path

import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer

ROOT = Path.home() / "locany"
DS = ROOT / "data/public647"
MODEL = ROOT / "models/LocateAnything-3B"
OUT = ROOT / "smoke"
OUT.mkdir(exist_ok=True)
CAM = "observation.images.camera_top"

# (episode, frame-in-episode, categories)
CASES = [
    (314, 0, ["mangosteen", "plate"]),
    (529, 0, ["beige lip gloss tube", "cosmetic organizer"]),
    (98, 0, ["paper ball", "brush", "dustpan", "basket"]),
]

info = json.load(open(DS / "meta/info.json"))
fps = info["fps"]
eps = pd.read_parquet(DS / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")


def frame(ep, j):
    r = eps.loc[ep]
    path = DS / info["video_path"].format(
        video_key=CAM,
        chunk_index=int(r[f"videos/{CAM}/chunk_index"]),
        file_index=int(r[f"videos/{CAM}/file_index"]),
    )
    vr = VideoReader(str(path), ctx=cpu(0))
    idx = int(round((r[f"videos/{CAM}/from_timestamp"] + j / fps) * fps))
    return Image.fromarray(vr[idx].asnumpy()).convert("RGB")


t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
print(f"load {time.time()-t0:.1f}s, mem {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


@torch.no_grad()
def detect(img, cats, mode="hybrid"):
    q = "Locate all the instances that matches the following description: " + "</c>".join(cats) + "."
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(msgs)
    inp = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    out = model.generate(
        pixel_values=inp["pixel_values"].to(torch.bfloat16),
        input_ids=inp["input_ids"],
        attention_mask=inp["attention_mask"],
        image_grid_hws=inp.get("image_grid_hws"),
        tokenizer=tok,
        max_new_tokens=2048,
        use_cache=True,
        generation_mode=mode,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        verbose=False,
    )
    return out[0] if isinstance(out, tuple) else out


for ep, j, cats in CASES:
    img = frame(ep, j)
    torch.cuda.synchronize(); t = time.time()
    ans = detect(img, cats)
    torch.cuda.synchronize(); dt = time.time() - t
    w, h = img.size
    boxes = [[int(g) for g in m.groups()] for m in BOX.finditer(ans)]
    print(f"\n== ep{ep} f{j} {w}x{h} {cats}  {dt:.2f}s  {len(boxes)} boxes")
    print(ans)
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], outline=(255, 0, 0), width=2)
    img.save(OUT / f"ep{ep:03d}_f{j}.jpg", quality=90)

print(f"\npeak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB allocated, "
      f"{torch.cuda.max_memory_reserved()/2**30:.2f} GiB reserved")
