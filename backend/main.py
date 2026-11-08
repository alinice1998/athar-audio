"""
FastAPI Server for Quranic Audio Style & Reverb Cloner
(خادم واجهة برمجة التطبيقات لاستوديو محاكاة الهندسة الصوتية للتلاوات القرآنية)
"""
import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch

from backend.dsp.pipeline import QuranicAudioPipeline
from backend.dsp.separator import AudioSeparatorEngine

app = FastAPI(
    title="استوديو محاكاة الهندسة الصوتية للتلاوات القرآنية",
    description="Quranic Audio Style & Reverb Cloner API",
    version="1.0.0"
)

# Enable CORS for local and web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
CACHE_DIR = os.path.join(STORAGE_DIR, "cache")
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Ensure sample files are synced to storage/samples if missing
SAMPLES_DIR = os.path.join(STORAGE_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
ROOT_SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
if os.path.exists(ROOT_SAMPLES_DIR):
    for fname in os.listdir(ROOT_SAMPLES_DIR):
        src_path = os.path.join(ROOT_SAMPLES_DIR, fname)
        dst_path = os.path.join(SAMPLES_DIR, fname)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                pass

# Initialize Pipeline
pipeline = QuranicAudioPipeline(sample_rate=44100, storage_dir=STORAGE_DIR)
separator_engine = AudioSeparatorEngine(cache_dir=CACHE_DIR)


@app.get("/api/samples")
def list_samples():
    """Returns pre-loaded Quranic audio samples for quick testing."""
    samples_dir = os.path.join(STORAGE_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    return {
        "references": [
            {
                "id": "sobhi_maryam",
                "name": "إسلام صبحي - سورة مريم (مؤثرات وصدى مسجدي مهيب)",
                "filename": "ref_islam_sobhi_maryam.wav",
                "url": "/audio/samples/ref_islam_sobhi_maryam.wav"
            },
            {
                "id": "sobhi_qaf",
                "name": "إسلام صبحي - سورة ق (مؤثرات واستوديو عميق)",
                "filename": "ref_islam_sobhi_qaf.wav",
                "url": "/audio/samples/ref_islam_sobhi_qaf.wav"
            },
            {
                "id": "alafasy",
                "name": "مشاري العفاسي - الفاتحة (صدى ناعم)",
                "filename": "ref_alafasy_fatiha.mp3",
                "url": "/audio/samples/ref_alafasy_fatiha.mp3"
            }
        ],
        "targets": [
            {
                "id": "badr_maryam",
                "name": "بدر التركي - سورة مريم (صوت جاف لنفس الآيات)",
                "filename": "target_dry_badr_turki_maryam.wav",
                "url": "/audio/samples/target_dry_badr_turki_maryam.wav"
            },
            {
                "id": "hazza_maryam",
                "name": "هزاع البلوشي - سورة مريم (صوت جاف لنفس الآيات)",
                "filename": "target_dry_hazza_maryam.wav",
                "url": "/audio/samples/target_dry_hazza_maryam.wav"
            },
            {
                "id": "badr_qaf",
                "name": "بدر التركي - سورة ق (صوت جاف لنفس الآيات)",
                "filename": "target_dry_badr_turki_qaf.wav",
                "url": "/audio/samples/target_dry_badr_turki_qaf.wav"
            }
        ]
    }


@app.get("/api/status")
def get_system_status():
    """Returns GPU and environment status."""
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "غير متوفر (استخدام المعالج CPU)"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if has_cuda else 0.0

    return {
        "status": "online",
        "cuda_available": has_cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "sample_rate": pipeline.sr
    }


@app.post("/api/check-cache")
async def check_reference_cache(ref_file: UploadFile = File(...)):
    """Checks if the reference file was previously separated and cached."""
    content = await ref_file.read()
    file_hash = separator_engine.compute_file_hash(content)
    entry_dir, dry_path, reverb_path = separator_engine.get_cached_paths(file_hash)
    is_cached = os.path.exists(dry_path) and os.path.exists(reverb_path)

    return {
        "hash": file_hash,
        "is_cached": is_cached,
        "filename": ref_file.filename,
        "size_bytes": len(content)
    }


@app.post("/api/process")
async def process_recitations(
    ref_file: Optional[UploadFile] = File(None),
    target_file: UploadFile = File(...),
    ref_cached_hash: Optional[str] = Form(None),
    eq_intensity: float = Form(0.8),
    eq_engine: str = Form("spectral"),
    reverb_mode: str = Form("parametric"),
    decay_scale: float = Form(1.0),
    reverb_mix: Optional[float] = Form(None),
    enable_ducking: bool = Form(True),
    ducking_amount_db: float = Form(2.5),
    clean_target: bool = Form(True)
):
    """
    Main audio processing endpoint.
    Processes reference recitation and target user recitation through the 5 DSP stages.
    """
    session_id = f"job_{uuid.uuid4().hex[:12]}"
    session_dir = os.path.join(OUTPUTS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    ref_audio_path = None

    # Handle Reference Audio
    if ref_file and ref_file.filename:
        ref_filename = f"reference_{ref_file.filename}"
        ref_audio_path = os.path.join(session_dir, ref_filename)
        with open(ref_audio_path, "wb") as f:
            shutil.copyfileobj(ref_file.file, f)
    elif ref_cached_hash:
        entry_dir, dry_path, reverb_path = separator_engine.get_cached_paths(ref_cached_hash)
        if os.path.exists(dry_path):
            ref_audio_path = dry_path  # Can use cached dry directly or cached files
        else:
            raise HTTPException(status_code=400, detail="بصمة التلاوة المرجعية المخبأة غير موجودة.")
    else:
        raise HTTPException(status_code=400, detail="يرجى إرفاق ملف التلاوة المرجعية.")

    # Handle Target User Audio
    target_filename = f"target_{target_file.filename or 'user_recording.wav'}"
    target_audio_path = os.path.join(session_dir, target_filename)
    with open(target_audio_path, "wb") as f:
        shutil.copyfileobj(target_file.file, f)

    try:
        result = pipeline.process(
            ref_file_path=ref_audio_path,
            target_file_path=target_audio_path,
            clean_target=clean_target,
            eq_intensity=eq_intensity,
            eq_engine=eq_engine,
            reverb_mode=reverb_mode,
            decay_scale=decay_scale,
            reverb_mix=reverb_mix,
            enable_ducking=enable_ducking,
            ducking_amount_db=ducking_amount_db,
            session_id=session_id
        )

        # Build public URLs
        base_audio_url = f"/audio/outputs/{session_id}"
        ref_hash = separator_engine.compute_file_hash(ref_audio_path)
        cache_audio_url = f"/audio/cache/{ref_hash}"

        return {
            "success": True,
            "session_id": session_id,
            "elapsed_seconds": result["elapsed_seconds"],
            "was_reference_cached": result["was_reference_cached"],
            "urls": {
                "final_master": f"{base_audio_url}/quranic_master.wav",
                "extracted_ir": f"{base_audio_url}/extracted_ir.wav",
                "target_raw": f"{base_audio_url}/{target_filename}",
                "target_clean": f"{base_audio_url}/target_dry_clean.wav",
                "target_eq": f"{base_audio_url}/target_eq_matched.wav",
                "reference": f"{base_audio_url}/{os.path.basename(ref_audio_path)}",
                "ref_dry": f"{cache_audio_url}/ref_dry.wav",
                "ref_reverb": f"{cache_audio_url}/ref_reverb.wav"
            },
            "parameters": result["parameters"],
            "acoustic_analysis": result["acoustic_analysis"],
            "spectrum_data": result["spectrum_data"],
            "mastering_stats": result["mastering_stats"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء معالجة الصوت: {str(e)}")


# Mount audio files directory
app.mount("/audio", StaticFiles(directory=STORAGE_DIR), name="audio")

# Mount frontend static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Quranic Audio Style & Reverb Cloner API is Running."}
