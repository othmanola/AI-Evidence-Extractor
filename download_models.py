from pathlib import Path
import shutil, json
import os
from huggingface_hub import hf_hub_download

MODEL_DIR = Path(os.environ.get(
    "AEE_DATA_DIR", str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor")
)) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SOURCES = [
    ("Subh775/Firearm_Detection_Yolov8n", "weights/best.pt", "firearm_yolov8n"),
    ("HaiderKhan6410/weapon-yolo26x", "model/best.pt", "weapon_yolo26x"),
    ("ayesha519/cypher-weapon-model", "best.pt", "cypher_weapon"),
    ("Shantanukadam/weapon_detection", "All_weapon.pt", "weapon_detection"),
    ("Hadi959/weapon-detection-yolov8", "best.pt", "weapon_yolov8"),
    ("iitolstykh/YOLO-Face-Person-Detector", "yolov8x_person_face.pt", "person_face"),
]
CONF = {"firearm_yolov8n":.75,"weapon_yolo26x":.15,"cypher_weapon":.15,"weapon_detection":.15,"weapon_yolov8":.15,"person_face":.25}
manifest=[]
for repo, filename, name in SOURCES:
    try:
        cached=hf_hub_download(repo, filename, repo_type="model")
        target=MODEL_DIR/f"{name}.pt"
        shutil.copy2(cached,target)
        manifest.append({"path":str(target),"source":repo,"name":name,"conf":CONF[name]})
        print(f"Ready: {name} ({target.stat().st_size/1e6:.1f} MB)")
    except Exception as exc:
        print(f"Skipped {name}: {type(exc).__name__}: {exc}")
(MODEL_DIR/"ensemble_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
if not manifest:
    raise RuntimeError("No detection models downloaded. Check the Colab internet connection and model repository availability.")
print(f"\n{len(manifest)} model(s) ready. The app will use the models that downloaded successfully.")


