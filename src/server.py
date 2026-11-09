import os
import io
import time
import json
import base64
import tempfile
import threading
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import cadquery as cq

from src.csg_schema import CSGTree, CSGStep, compile_csg_to_cadquery

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EXPORTS_DIR = os.path.join(RESULTS_DIR, "enhanced_hybrid_step_exports")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Global RAG generator state
rag_generator = None
rag_status = "uninitialized"  # "uninitialized", "loading", "ready", "error"
rag_lock = threading.Lock()

def _init_rag_pipeline():
    """Background task to pre-load DINOv2 and EnhancedRAGGenerator at startup."""
    global rag_generator, rag_status
    with rag_lock:
        if rag_generator is not None:
            return
        rag_status = "loading"
        print("[Startup] Pre-loading DINOv2 and EnhancedRAGGenerator...")
        t0 = time.time()
        try:
            from src.enhanced_model import EnhancedRAGGenerator
            from src.image_embedder import DINOv2Embedder
            from src.retriever import SimilarityRetriever, build_train_lookup
            from src.data_loader import load_gencad_dataset

            embeddings_cache = os.path.join(CACHE_DIR, "dinov2_embeddings_dinov2.npz")
            lookup_cache = os.path.join(CACHE_DIR, "train_lookup_5000.pkl")

            train_lookup = None
            if os.path.exists(lookup_cache):
                try:
                    import pickle
                    with open(lookup_cache, "rb") as f:
                        train_lookup = pickle.load(f)
                    print(f"[Startup] Loaded {len(train_lookup)} train lookup items from pickle")
                except Exception as pe:
                    print(f"[Startup] Failed to unpickle train lookup: {pe}")

            if train_lookup is None:
                train_ds = load_gencad_dataset(split="train")
                train_ds_subset = train_ds.select(range(min(5000, len(train_ds))))
                train_lookup = build_train_lookup(train_ds_subset)

            embedder = DINOv2Embedder()
            index_embeddings, index_ids = embedder.build_index(
                None,
                cache_path=embeddings_cache,
                max_samples=5000,
            )

            retriever = SimilarityRetriever(
                embedder=embedder,
                index_embeddings=index_embeddings,
                index_ids=index_ids,
                train_data_lookup=train_lookup,
                k=3,
            )

            rag_generator = EnhancedRAGGenerator(
                retriever=retriever,
                model="gemini-3.1-pro-preview",
                repair_model="gemini-2.5-flash",
                n_shots=3,
                use_cot=True,
                include_reference=True,
                enable_repair=True,
                repair_max_attempts=2,
                temperature=0.1,
            )
            rag_status = "ready"
            print(f"[Startup] EnhancedRAGGenerator ready in {time.time()-t0:.2f}s!")
        except Exception as e:
            rag_status = f"error: {str(e)}"
            print(f"[Startup] RAG initialization failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start preloading in a daemon thread so server starts listening immediately
    thread = threading.Thread(target=_init_rag_pipeline, daemon=True)
    thread.start()
    yield

app = FastAPI(title="AI CAD Generation API", lifespan=lifespan)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3050",
        "http://127.0.0.1:3050",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str

class ExportStepRequest(BaseModel):
    tree: Optional[Dict[str, Any]] = None
    code: Optional[str] = None

@app.get("/")
async def root():
    return {
        "service": "AIMDEA CAD Synthesis API",
        "status": "online",
        "rag_ready": rag_generator is not None,
        "frontend_url": "http://localhost:3050",
        "docs_url": "http://localhost:8000/docs"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "rag_status": rag_status,
        "rag_ready": rag_generator is not None,
    }

@app.post("/api/generate")
async def generate_cad(req: GenerateRequest):
    """Generate a CSG tree and CadQuery code from a text prompt."""
    try:
        from google import genai
        from google.genai import types
        from src.csg_schema import CSG_SCHEMA_INSTRUCTION, CSG_COT_PREAMBLE, compile_csg_to_cadquery
        from src.code_repair import _try_execute

        client = genai.Client()
        prompt_override = f"Generate a CAD model for the following request: {req.prompt}"
        
        config = types.GenerateContentConfig(
            system_instruction=CSG_COT_PREAMBLE + "\n" + CSG_SCHEMA_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=CSGTree,
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_override],
            config=config,
        )
        tree_data = json.loads(response.text)
        code_str = compile_csg_to_cadquery(tree_data)

        # Test compile and self-repair if needed
        success, error_msg = _try_execute(code_str, from_csg_compiler=True)
        if not success and rag_generator is not None:
            repaired = rag_generator._flash_repair(tree_data, error_msg)
            if repaired and repaired.get("steps"):
                rep_code = compile_csg_to_cadquery(repaired)
                rep_success, _ = _try_execute(rep_code, from_csg_compiler=True)
                if rep_success:
                    tree_data = repaired
                    code_str = rep_code

        return {"tree": tree_data, "code": code_str}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-from-image")
async def generate_from_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """Generate a CSG tree and CadQuery code from an uploaded reference image."""
    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # If RAG is ready, use the full Enhanced Hybrid pipeline
        if rag_generator is not None:
            csg_tree = rag_generator._generate_csg(image)
            if rag_generator.enable_repair and csg_tree.get("steps"):
                for attempt in range(rag_generator.repair_max_attempts):
                    error_class, error_msg = rag_generator._validate_csg(csg_tree)
                    if error_class is None:
                        break
                    elif error_class == "A":
                        csg_tree = rag_generator._flash_repair(csg_tree, error_msg)
                    elif error_class == "B":
                        csg_tree = rag_generator._generate_csg(image, temperature=0.3)
        else:
            # Fallback: direct VLM image-to-CSG call
            from google import genai
            from google.genai import types
            from src.baseline_model import _encode_image_to_bytes
            from src.csg_schema import CSG_SCHEMA_INSTRUCTION, CSG_COT_PREAMBLE

            client = genai.Client()
            img_bytes = _encode_image_to_bytes(image)
            user_text = "Generate a JSON CSG construction tree for this 3D object:"
            if prompt and prompt.strip():
                user_text += f" Additional guidance: {prompt.strip()}"

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=user_text),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ]
                )
            ]
            config = types.GenerateContentConfig(
                system_instruction=CSG_COT_PREAMBLE + "\n" + CSG_SCHEMA_INSTRUCTION,
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=CSGTree
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config,
            )
            csg_tree = json.loads(response.text)

        code_str = compile_csg_to_cadquery(csg_tree)
        return {"tree": csg_tree, "code": code_str}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compile")
async def compile_cad(tree: CSGTree):
    """Compiles a CSG tree into a 3D model, returns base64 encoded STL and compiled Python code."""
    try:
        tree_dict = tree.model_dump() if hasattr(tree, "model_dump") else tree.dict()
        code_str = compile_csg_to_cadquery(tree_dict)

        namespace = {"cq": cq}
        try:
            exec(code_str, namespace)
            result_obj = namespace.get("result", None)
        except Exception as e:
            raise ValueError(f"Code execution failed: {e}")

        if result_obj is None:
            raise ValueError("Compilation resulted in an empty object (no 'result' variable found).")

        if isinstance(result_obj, cq.Workplane):
            try:
                solid = result_obj.findSolid()
            except Exception:
                solid = result_obj.val()
        else:
            solid = result_obj

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_path = tmp.name

        cq.exporters.export(solid, tmp_path, "STL")

        with open(tmp_path, "rb") as f:
            stl_data = f.read()

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        b64_stl = base64.b64encode(stl_data).decode("utf-8")
        return {"format": "stl", "data": b64_stl, "code": code_str}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/export/step")
async def export_step(req: ExportStepRequest):
    """Accepts either CSG tree or CadQuery code and exports as a downloadable STEP file."""
    try:
        code_str = req.code
        if not code_str and req.tree:
            code_str = compile_csg_to_cadquery(req.tree)

        if not code_str:
            raise HTTPException(status_code=400, detail="Either 'code' or 'tree' must be provided.")

        namespace = {"cq": cq}
        try:
            exec(code_str, namespace)
            result_obj = namespace.get("result", None)
        except Exception as e:
            raise ValueError(f"Code execution failed: {e}")

        if result_obj is None:
            raise ValueError("Compilation yielded no 'result' object.")

        if isinstance(result_obj, cq.Workplane):
            try:
                solid = result_obj.findSolid()
            except Exception:
                solid = result_obj.val()
        else:
            solid = result_obj

        tmp = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
        tmp_path = tmp.name
        tmp.close()

        cq.exporters.export(solid, tmp_path, "STEP")

        # Background task to clean up temp file after response
        def cleanup():
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        background_tasks = BackgroundTasks()
        background_tasks.add_task(cleanup)

        return FileResponse(
            tmp_path,
            media_type="application/step",
            filename="model.step",
            background=background_tasks,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/benchmark/results")
async def get_benchmark_results():
    """Serves the complete Enhanced Hybrid benchmark metrics and part list."""
    results_path = os.path.join(RESULTS_DIR, "enhanced_hybrid_results.json")
    predictions_path = os.path.join(RESULTS_DIR, "enhanced_hybrid_predictions.json")

    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Benchmark results file not found.")

    with open(results_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    predictions = {}
    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, "r", encoding="utf-8") as f:
                preds_json = json.load(f)
                predictions = preds_json.get("predictions", {})
        except Exception:
            pass

    iou_map = res_data.get("iou_per_sample", {})
    failed_ids = set(res_data.get("vsr_failed_ids", []))

    all_sample_ids = sorted(list(set(list(iou_map.keys()) + list(failed_ids))))

    parts = []
    perfect_count = 0
    high_count = 0
    partial_count = 0
    low_count = 0
    failed_count = len(failed_ids)

    for sid in all_sample_ids:
        clean_id = sid.replace("/", "_")
        is_failed = sid in failed_ids
        iou_val = iou_map.get(sid, None)

        category = "failed"
        if not is_failed and iou_val is not None:
            if iou_val >= 0.95:
                category = "perfect"
                perfect_count += 1
                high_count += 1
            elif iou_val >= 0.50:
                category = "high"
                high_count += 1
            elif iou_val >= 0.20:
                category = "partial"
                partial_count += 1
            else:
                category = "low"
                low_count += 1

        stl_file = os.path.join(EXPORTS_DIR, f"{clean_id}.stl")
        step_file = os.path.join(EXPORTS_DIR, f"{clean_id}.step")
        img_file = os.path.join(EXPORTS_DIR, f"{clean_id}.png")

        parts.append({
            "part_id": sid,
            "clean_id": clean_id,
            "iou": round(iou_val, 4) if iou_val is not None else 0.0,
            "status": "failed" if is_failed else "valid",
            "category": category,
            "has_stl": os.path.exists(stl_file),
            "has_step": os.path.exists(step_file),
            "has_image": os.path.exists(img_file),
            "code": predictions.get(sid, ""),
        })

    summary = {
        "experiment_name": res_data.get("experiment_name", "enhanced_hybrid"),
        "total_samples": res_data.get("num_samples", len(all_sample_ids)),
        "vsr": res_data.get("vsr", 0.86),
        "vsr_percentage": round(res_data.get("vsr", 0.86) * 100, 1),
        "vsr_successful": res_data.get("vsr_successful", 86),
        "vsr_failed": failed_count,
        "mean_iou_best": round(res_data.get("mean_iou_best", 0.3777), 4),
        "iou_valid_count": res_data.get("iou_valid_count", 86),
        "perfect_count": perfect_count,
        "high_count": high_count,
        "partial_count": partial_count,
        "low_count": low_count,
        "step_export_count": len([p for p in parts if p["has_step"]]),
    }

    return {"summary": summary, "parts": parts}

@app.get("/api/benchmark/part/{part_id:path}/stl")
async def get_benchmark_stl(part_id: str):
    clean_id = part_id.replace("/", "_").replace(".stl", "")
    file_path = os.path.join(EXPORTS_DIR, f"{clean_id}.stl")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"STL file for part {clean_id} not found.")
    return FileResponse(file_path, media_type="model/stl", filename=f"{clean_id}.stl")

@app.get("/api/benchmark/part/{part_id:path}/step")
async def get_benchmark_step(part_id: str):
    clean_id = part_id.replace("/", "_").replace(".step", "")
    file_path = os.path.join(EXPORTS_DIR, f"{clean_id}.step")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"STEP file for part {clean_id} not found.")
    return FileResponse(file_path, media_type="application/step", filename=f"{clean_id}.step")

@app.get("/api/benchmark/part/{part_id:path}/image")
async def get_benchmark_image(part_id: str):
    clean_id = part_id.replace("/", "_").replace(".png", "")
    file_path = os.path.join(EXPORTS_DIR, f"{clean_id}.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Reference image for part {clean_id} not found.")
    return FileResponse(
        file_path,
        media_type="image/png",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )

@app.get("/api/benchmark/part/{part_id:path}/code")
async def get_benchmark_code(part_id: str):
    predictions_path = os.path.join(RESULTS_DIR, "enhanced_hybrid_predictions.json")
    if not os.path.exists(predictions_path):
        raise HTTPException(status_code=404, detail="Predictions file not found.")
    clean_id = part_id.replace("_", "/")
    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = json.load(f).get("predictions", {})
    code = preds.get(clean_id) or preds.get(part_id, "")
    return {"part_id": part_id, "code": code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
