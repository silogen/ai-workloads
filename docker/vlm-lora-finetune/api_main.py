# main.py
import asyncio
import io
import os
import re
import shutil
import time
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

import job_db
import merge_and_infer
import yaml  # type: ignore[import-untyped]
from clipora.config import TrainConfig, parse_yaml_to_config
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from train import main as train_main

# Folder where user uploaded files like training data zips will be stored
FILE_DOWNLOAD_DIR = os.getenv("FILE_DOWNLOAD_PATH", "/tmp/downloaded_data/")
# base folder for training jobs, each job will have its own job id folder
TRAIN_JOB_OUTPUT_DIR = os.getenv("TRAIN_JOB_OUTPUT_DIR", "/tmp/trained_models/")

os.makedirs(FILE_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TRAIN_JOB_OUTPUT_DIR, exist_ok=True)


# --- Lifespan Manager for Executor and DB ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: Create process pool and initialize the database
    # will be used to run CPU-intensive tasks in a separate process
    # so that API will stay responsive
    job_db.init_db()
    app.state.process_pool = ProcessPoolExecutor()
    yield
    # On shutdown: Gracefully close the process pool
    app.state.process_pool.shutdown()


# --- (Modified) Background Task (runs in a separate process) ---
def train_job(job_id: str, config: TrainConfig, zip_path: str | None = None):
    """
    This function is CPU-intensive and runs in a separate process.
    It communicates status by calling functions from the `db` module.
    """
    # Create job specific output directory for training data
    job_training_data_dir = os.path.join(TRAIN_JOB_OUTPUT_DIR, job_id, "training_data")
    os.makedirs(job_training_data_dir, exist_ok=True)
    if zip_path:
        try:
            job_db.update_job(job_id, status="extracting", detail=f"Extracting {os.path.basename(zip_path)}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(job_training_data_dir)
        except Exception as e:
            job_db.update_job(job_id, status="failed", detail=str(e))
            return

        # Change some paths in the config to point to the extracted data path
        config.train_dataset = os.path.join(job_training_data_dir, os.path.basename(config.train_dataset))
        config.eval_dataset = os.path.join(job_training_data_dir, os.path.basename(config.eval_dataset))

    try:
        # output trained loras to job specific output dir
        config.output_dir = os.path.join(TRAIN_JOB_OUTPUT_DIR, job_id)

        job_db.update_job(job_id, status="training", detail="Model training in progress...")
        job_callback = job_db.create_job_callback(job_id)
        train_main(config, job_callback)

        job_db.update_job(job_id, status="complete", detail="Training finished successfully.")
    except Exception as e:
        job_db.update_job(job_id, status="failed", detail=str(e))


# --- FastAPI App ---
app = FastAPI(
    title="Training API",
    description="An API to submit training jobs with status tracking via SQLite.",
    version="2.1.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",  # If you serve the html with `python -m http.server`
    "null",  # Allow requests from local files (i.e., opening the HTML with file://)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Training Endpoint ---
@app.post("/train/", summary="Submit a training job", status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    config_str: str = Form(..., description="A YAML string for training config."),
    file: Optional[UploadFile] = File(
        None, description="A ZIP file with the dataset (required if not using HuggingFace datasets)."
    ),
):
    job_id = str(uuid.uuid4())

    try:
        config_dict = yaml.safe_load(config_str)
        config = TrainConfig(**config_dict)
    except yaml.YAMLError:
        raise HTTPException(status_code=400, detail="Config is not valid YAML.")

    # Check if file is required based on config.datatype
    if config.datatype != "hf":
        if file is None:
            raise HTTPException(
                status_code=400,
                detail="Custom data requires a ZIP file upload. Set 'datatype' to 'hf' to use HuggingFace datasets.",
            )
        if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Expected ZIP.")

        zip_path = os.path.join(FILE_DOWNLOAD_DIR, f"{job_id}.zip")
        try:
            with open(zip_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    buffer.write(chunk)
        finally:
            await file.close()
    else:
        zip_path = None  # Not needed for HuggingFace datasets

    job_db.create_job(job_id)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(app.state.process_pool, train_job, job_id, config, zip_path)

    return {
        "message": "Training job accepted.",
        "job_id": job_id,
        "status_url": app.url_path_for("get_status", job_id=job_id),
    }


def _extract_percentage(s):
    match = re.search(r"(\d+\.?\d*)%", s)
    return float(match.group(1)) if match else None


def _job_status_return(job_dict):
    job_dict["progress"] = _extract_percentage(job_dict["detail"]) if job_dict["detail"] else None
    return job_dict


# --- Job Status Endpoint ---
@app.get("/status/{job_id}", summary="Get job status", name="get_status")
async def get_status(job_id: str):
    """
    Polls the database to get the current status of the training job.
    """

    job = job_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _job_status_return(job)


@app.get("/list_jobs/", summary="Return all jobs", name="list_jobs")
async def list_jobs():
    """
    Fetches all jobs from the database.
    """

    jobs = job_db.get_all_jobs()
    if not jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No jobs found.")
    return [_job_status_return(job) for job in jobs]


@app.post("/inference/", summary="Run inference on a single image with specified classes", name="classes_inference")
async def classes_inference(job_id: str, image: UploadFile, classes: str = Form(...)):
    """
    Accepts an image and a list of class names from a form submission,
    processes them, and returns the details.

    - **image**: The uploaded image file.
    - **classes**: A list of strings separated by new lines, used as texts/"classes".
    """

    classes_list = [line.strip() for line in classes.split("\n") if line.strip()]
    image_path = os.path.join("/tmp/", str(uuid.uuid4()) + "_" + image.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    job_dict = job_db.get_job(job_id)
    if job_dict is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")

    probabilities, classes_list = merge_and_infer.run_single_inference(job_dict, image_path, classes_list)
    if probabilities is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Failed to run inference.")
    return {"probabilities": probabilities, "classes": classes_list}


@app.get("/download_finetuned_model/{job_id}")
async def download_finetuned_model(job_id: str):
    """
    Packages the contents of a specified model folder into a ZIP file
    and returns it for download.
    """
    job = job_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found in database.")
    model_folder_path = job["best_finetuned_model_path"]
    # if relative path, use TRAIN_JOB_OUTPUT_DIR/job_id/model_folder_path as base
    if not os.path.isabs(model_folder_path):
        model_folder_path = os.path.join(TRAIN_JOB_OUTPUT_DIR, job_id, model_folder_path)
    # 2. Validate that the requested directory exists and is actually a directory.
    if not os.path.isdir(model_folder_path):
        raise HTTPException(
            status_code=404, detail=f"Job ID '{job_id}' does not have a valid model folder at '{model_folder_path}'."
        )

    # 3. Create an in-memory binary stream (a virtual file) to hold the ZIP data.
    # This avoids writing a temporary file to the disk, which is more efficient.
    zip_io_buffer = io.BytesIO()

    # 4. Create the ZIP file within the in-memory buffer.
    with zipfile.ZipFile(zip_io_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as temp_zip_file:
        for root, _, files in os.walk(model_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Define the name of the file inside the ZIP archive.
                # os.path.relpath ensures the paths are relative to the model folder,
                # recreating the directory structure correctly inside the zip.
                archive_name = os.path.relpath(file_path, model_folder_path)
                temp_zip_file.write(file_path, arcname=archive_name)

    # 5. Rewind the in-memory buffer to the beginning.
    zip_io_buffer.seek(0)
    download_filename = f"{job_id}_model.zip"

    return StreamingResponse(
        content=zip_io_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={download_filename}"},
    )


def extract_zip_from_path_sync(zip_path: str, output_dir: str):
    """Synchronously extracts a zip file from a path to a directory."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


@app.post("/upload_finetuned_lora/", summary="Upload a pre-trained LoRA model", status_code=status.HTTP_202_ACCEPTED)
async def upload_finetuned_lora(
    file: UploadFile = File(..., description="A ZIP file containing the finetuned LoRA model artifacts.")
):
    """
    Accepts a ZIP file containing a pre-trained model, extracts it,
    and registers it as a completed job. This allows for using the model
    for inference without running the training process through this API.

    ZIP file needs to have the model weights and config files, including train_config.yaml.
    train_config.yaml is needed so we know the base model
    """
    job_id = str(uuid.uuid4())
    job_db.create_job(job_id, status="uploading", detail="Receiving LoRA model file.")

    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        detail = "Invalid file type. Expected a ZIP file."
        job_db.update_job(job_id, status="failed", detail=detail)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    model_folder_name = "uploaded_model"
    job_uploaded_model_dir = os.path.join(job_id, model_folder_name)
    output_dir = os.path.join(TRAIN_JOB_OUTPUT_DIR, job_uploaded_model_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        job_db.update_job(job_id, status="extracting", detail="Extracting model artifacts from ZIP file.")

        zip_path = os.path.join(FILE_DOWNLOAD_DIR, f"{job_id}.zip")
        with open(zip_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        await run_in_threadpool(extract_zip_from_path_sync, zip_path, output_dir)

        job_db.update_job(
            job_id,
            status="complete",
            detail="LoRA model successfully uploaded and registered.",
            # save with the relative path in case job output dir changes later
            best_finetuned_model_path=model_folder_name,
        )
    except Exception as e:
        job_db.update_job(job_id, status="failed", detail=f"Failed to process ZIP file: {e}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the file: {e}",
        )
    finally:
        await file.close()

    return {
        "message": "Finetuned LoRA model uploaded and registered successfully.",
        "job_id": job_id,
        "status_url": app.url_path_for("get_status", job_id=job_id),
    }


@app.get("/")
def read_root():
    return {"message": "Welcome to the Training API. POST to /train/ to submit a job."}
