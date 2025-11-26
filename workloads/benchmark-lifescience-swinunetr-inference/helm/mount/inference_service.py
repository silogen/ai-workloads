import argparse
import logging
import os
import tempfile
import traceback
from contextlib import asynccontextmanager, nullcontext

import nibabel as nib
import numpy as np
import torch
import uvicorn
from data_utils import DEVICE, IMAGE_DATA, get_post_transforms_inverter, get_transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from monai import transforms
from swinunetr import SwinUnetrModelForInference
from torch.cuda.amp import autocast

LOGGER = logging.getLogger(__name__)

# Read configuration from environment variables with fallbacks
ARGS = argparse.Namespace(
    hf_model=os.environ.get("HF_MODEL", "darragh/swinunetr-btcv-base"),
    roi_x=int(os.environ.get("ROI_X", "96")),
    roi_y=int(os.environ.get("ROI_Y", "96")),
    roi_z=int(os.environ.get("ROI_Z", "96")),
    space_x=float(os.environ.get("SPACE_X", "1.5")),
    space_y=float(os.environ.get("SPACE_Y", "1.5")),
    space_z=float(os.environ.get("SPACE_Z", "2.0")),
    a_min=float(os.environ.get("A_MIN", "-175.0")),
    a_max=float(os.environ.get("A_MAX", "250.0")),
    b_min=float(os.environ.get("B_MIN", "0.0")),
    b_max=float(os.environ.get("B_MAX", "1.0")),
    infer_overlap=float(os.environ.get("INFER_OVERLAP", "0.5")),
    compile=os.environ.get("COMPILE", "false").lower() == "true",
    compile_mode=os.environ.get("COMPILE_MODE", "max-autotune"),
    autocast=os.environ.get("AUTOCAST", "false").lower() == "true",
)

MODEL: SwinUnetrModelForInference | None = None
IMAGE_TRANSFORMS: transforms.Compose | None = None
POST_TRANSFORMS_INVERTER: transforms.Compose | None = None
AUTOCAST_CONTEXT = nullcontext()


def load_model(args):
    """Loads the pre-trained model."""
    global MODEL, IMAGE_TRANSFORMS, POST_TRANSFORMS_INVERTER, AUTOCAST_CONTEXT
    MODEL = SwinUnetrModelForInference.from_pretrained(args.hf_model)

    if args.compile:
        LOGGER.info(f"Compiling model to {args.compile_mode}")
        MODEL = torch.compile(MODEL, mode=args.compile_mode, dynamic=False)
        LOGGER.info("Model compiled successfully.")

    # Set the model to evaluation mode
    MODEL.eval()
    MODEL.to(DEVICE)
    LOGGER.info(f"Model loaded on {DEVICE}.")

    # Configure autocast context once during model loading
    if args.autocast:
        LOGGER.info("Configuring autocast for inference")
        AUTOCAST_CONTEXT = autocast()
    else:
        AUTOCAST_CONTEXT = nullcontext()

    IMAGE_TRANSFORMS = get_transforms(args=args)
    POST_TRANSFORMS_INVERTER = get_post_transforms_inverter(IMAGE_TRANSFORMS)
    LOGGER.info("Image transform pipeline initialized.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on server startup."""
    LOGGER.info("Server startup: Loading model...")
    global MODEL, IMAGE_TRANSFORMS, POST_TRANSFORMS_INVERTER, AUTOCAST_CONTEXT
    try:
        load_model(ARGS)
    except Exception as e:
        LOGGER.error(f"Error loading model during startup: {e}")
        LOGGER.error(f"Traceback: \n{traceback.format_exc()}")
    yield

    MODEL = None
    IMAGE_TRANSFORMS = None
    POST_TRANSFORMS_INVERTER = None
    AUTOCAST_CONTEXT = nullcontext()


app = FastAPI(title="SwinUNETR Inference Service", lifespan=lifespan)


@app.get("/health", status_code=200)
async def health_check():
    """
    Simple health check endpoint.
    Returns 200 OK if the model is loaded and the server is running.
    """
    if MODEL is None or IMAGE_TRANSFORMS is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Server is not ready yet."  # 503 Service Unavailable
        )
    return {"status": "ok"}


@app.post("/predict/", response_class=FileResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file (NIfTI or .npy), performs inference,
    and returns the segmentation mask as a NIfTI file.
    """
    if MODEL is None or IMAGE_TRANSFORMS is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Server might be starting or encountered an error."
        )

    try:
        original_filename = file.filename
        file_suffix = ""
        if original_filename.endswith(".nii.gz"):
            file_suffix = ".nii.gz"
        elif original_filename.endswith(".nii"):
            file_suffix = ".nii"
        elif original_filename.endswith(".npy"):
            file_suffix = ".npy"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .nii, .nii.gz, or .npy")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        LOGGER.info(f"Temporary file saved at: {tmp_file_path} for original: {original_filename}")

        original_affine = np.eye(4)  # Default for npy
        original_shape_3d = None
        img_data_np = None
        if file_suffix in [".nii", ".nii.gz"]:
            nib_image = nib.load(tmp_file_path)
            img_data_np = np.asarray(nib_image.dataobj, dtype=np.float32)
            original_affine = nib_image.affine.astype(np.float32)
            original_shape_3d = img_data_np.shape
        elif file_suffix == ".npy":
            img_data_np = np.load(tmp_file_path).astype(np.float32)
            original_shape_3d = img_data_np.shape
            LOGGER.warning(
                "Using identity affine for .npy input. Inverse transform might not perfectly restore original physical space if original .npy had specific spacing."
            )

        # Prepare data dictionary for MONAI transforms
        data_dict = {IMAGE_DATA: tmp_file_path}
        # Apply transforms
        val_input_transformed_dict = IMAGE_TRANSFORMS(data_dict)

        val_inputs = val_input_transformed_dict[IMAGE_DATA]
        val_inputs = val_inputs.unsqueeze(0).to(DEVICE)
        LOGGER.info(f"Shape transformed inputs: {val_inputs.shape}")

        LOGGER.info("Predicting...")
        with torch.inference_mode(), AUTOCAST_CONTEXT:
            logits = MODEL.forward(
                inputs=val_inputs,
                roi_size=(ARGS.roi_x, ARGS.roi_y, ARGS.roi_z),
                sw_batch_size=4,
                overlap=ARGS.infer_overlap,
                mode="gaussian",
            )
            LOGGER.info(f"Shape of prediction before inversion: {logits.shape}")

        # Post-processing and transforms inversion
        invert_dict = {
            IMAGE_DATA: logits[0],
        }
        inverted = POST_TRANSFORMS_INVERTER(invert_dict)
        inverted_logits = inverted[IMAGE_DATA]
        LOGGER.info(f"Shape of prediction after inverse transforms: {inverted_logits.shape}")

        probs = torch.sigmoid(inverted_logits)
        LOGGER.info(f"Max/Min probs: {probs.max()} / {probs.min()}")
        seg = (probs > 0.5).float()
        prediction_np = seg.cpu().numpy().squeeze()
        LOGGER.info(f"Final prediction shape: {prediction_np.shape}")

        # Save prediction to a NIfTI file
        output_affine = inverted[IMAGE_DATA].affine.cpu().numpy()
        pred_nifti = nib.Nifti1Image(prediction_np, output_affine)

        with tempfile.NamedTemporaryFile(delete=False, suffix="_prediction.nii.gz") as pred_output_file:
            nib.save(pred_nifti, pred_output_file.name)
            response_file_path = pred_output_file.name

        os.unlink(tmp_file_path)

        return FileResponse(response_file_path, media_type="application/gzip", filename="prediction.nii.gz")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        if "tmp_file_path" in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        if "response_file_path" in locals() and os.path.exists(response_file_path):
            os.unlink(response_file_path)
        LOGGER.error(f"Error during prediction: {e}")
        LOGGER.error(f"Traceback: \n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            LOGGER.info(f"Temporary input file {tmp_file_path} unlinked.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
