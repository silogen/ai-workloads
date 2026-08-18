import copy
import json
import logging
import os
import tempfile
import warnings
import zipfile

import annotators
import imageio.v2 as imageio
import torch
from annotators.utils import read_video_frames, save_one_video
from configs import VACE_PREPROCCESS_CONFIGS
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class VACEHandler(BaseHandler):
    """
    Simplified TorchServe handler that invokes only VACE inference CLI.
    Requires local VACE clone; set VACE_REPO_PATH accordingly.
    """

    def initialize(self, ctx):
        """In this initialize function, the VACE model is
            loaded and initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        # Path to local VACE repo
        self.vace_repo = os.environ.get("VACE_REPO_DIR", "")

        # Create temp directory that persists during handler lifetime
        self.temp_dir = tempfile.mkdtemp()

        # Manifest and model directory
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # Diffusers model directory
        DIFFUSERS_DIR = model_dir + "/diffusers_model"

        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(DIFFUSERS_DIR)

        self.pipe = DiffusionPipeline.from_pretrained(DIFFUSERS_DIR)
        self.pipe.to(self.device)
        logger.info("Diffusion model from path %s loaded successfully", DIFFUSERS_DIR)

        self.initialized = True

    def preprocess(self, data):
        record = data[0]
        payload = record.get("body", record)

        if type(payload) is bytearray:
            payload = json.loads(payload.decode("utf-8"))
        task = payload.get("task", "")

        # Create per-request workspace
        workdir = tempfile.mkdtemp(dir=self.temp_dir)
        logging.info(f"Created temporary workspace: {workdir}")

        if payload.get("masked_video"):
            pass
        elif task:
            ALLOWED_TASKS = ["outpainting", "inpainting"]
            if task not in ALLOWED_TASKS:
                raise ValueError(f"Task {task} not supported. Allowed tasks: {ALLOWED_TASKS}")

            bbox = payload.get("bbox", None)
            caption = payload.get("caption", None)
            label = payload.get("label", None)
            save_fps = payload.get("save_fps", None)

            if task in ["inpainting"]:
                mode = payload.get("mode", "salientmasktrack")
                logging.info(f"Using inpainting mode: {mode}")

            task_cfg = copy.deepcopy(VACE_PREPROCCESS_CONFIGS)[task]
            class_name = task_cfg.pop("NAME")
            input_params = task_cfg.pop("INPUTS")
            output_params = task_cfg.pop("OUTPUTS")

            # Load video data
            video_bytes = record.get("video")

            if not video_bytes:
                raise KeyError(f"Missing uploaded video file for task {task}")
            # Set video file path
            video_path = os.path.join(workdir, "input_video.mp4")
            # Write video to file
            with open(video_path, "wb") as vf:
                vf.write(video_bytes)

            # input data
            fps = None
            input_data = copy.deepcopy(input_params)
            if "video" in input_params:
                if video_path is None:
                    raise ValueError("Missing input video path. Please provide a valid video file.")
                frames, fps, width, height, num_frames = read_video_frames(
                    video_path.split(",")[0], use_type="cv2", info=True
                )
                if frames is None:
                    raise ValueError("Video read error")
                input_data["frames"] = frames
                input_data["video"] = video_path
            if "frames" in input_params:
                if video_path is None:
                    raise ValueError("Please set video or check configs")
                frames, fps, width, height, num_frames = read_video_frames(
                    video_path.split(",")[0], use_type="cv2", info=True
                )
                if frames is None:
                    raise ValueError("Video read error")
                input_data["frames"] = frames
            if "bbox" in input_params:
                if bbox is not None:
                    input_data["bbox"] = bbox[0] if len(bbox) == 1 else bbox
            if "label" in input_params:
                input_data["label"] = label.split(",") if label is not None else None
            if "caption" in input_params:
                input_data["caption"] = caption
            if "mode" in input_params:
                input_data["mode"] = mode
            if "direction" in input_params:
                if payload.get("direction") is not None:
                    input_data["direction"] = payload.get("direction", "up,down,left,right").split(",")
            if "expand_ratio" in input_params:
                if payload.get("expand_ratio") is not None:
                    input_data["expand_ratio"] = payload.get("expand_ratio", "0.8")

            # Modify paths in `task_cfg` to become absolute and work with the handler
            for key, value in task_cfg.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and (sub_value.startswith("models/") or "/" in sub_value):
                            value[sub_key] = f"{self.vace_repo}/{sub_value}"
                elif isinstance(value, str) and (value.startswith("models/") or "/" in value):
                    task_cfg[key] = f"{self.vace_repo}/{value}"

            logging.info(f"Task config: {task_cfg}")

            cwd = os.getcwd()
            logging.info(f"Current working directory: {cwd}")
            logging.info(f"Changing working directory to {self.vace_repo}")
            try:
                os.chdir(self.vace_repo)  # make 'models/...' resolve to '/VACE/models/...'
                pre_ins = getattr(annotators, class_name)(cfg=task_cfg, device=f"cuda:{os.getenv('RANK', 0)}")
            finally:
                logging.info(f"Reverting working directory to {cwd}")
                os.chdir(cwd)
            results = pre_ins.forward(**input_data)

            # output data
            save_fps = fps if fps is not None else save_fps
            pre_save_dir = workdir
            if not os.path.exists(pre_save_dir):
                os.makedirs(pre_save_dir)

            out_data = {}
            if "frames" in output_params:
                frames = results["frames"] if isinstance(results, dict) else results
                if frames is not None:
                    save_path = os.path.join(pre_save_dir, f"src_video-{task}.mp4")
                    save_one_video(save_path, frames, fps=save_fps)
                    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                        logging.info(f"Saved frames result to {save_path}")
                        out_data["src_video"] = save_path
                    else:
                        logging.error(f"Failed to save video to {save_path}")
                        out_data["error_message"] = "Source video could not be generated"
                        out_data["error_status"] = True
                        out_data["src_video"] = save_path
            if "masks" in output_params:
                frames = results["masks"] if isinstance(results, dict) else results
                if frames is not None:
                    save_path = os.path.join(pre_save_dir, f"src_mask-{task}.mp4")
                    save_one_video(save_path, frames, fps=save_fps)
                    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                        logging.info(f"Saved frames result to {save_path}")
                        out_data["src_mask"] = save_path
                    else:
                        logging.error(f"Failed to save video to {save_path}")
                        out_data["error_message"] = "Mask video could not be generated"
                        out_data["error_status"] = True
                        out_data["src_mask"] = save_path

            # masked inference
            out_data["prompt"] = payload.get("prompt", "")
            out_data["src_ref_images"] = None
            out_data["model_name"] = os.getenv("MODEL_NAME", "Wan2.1-VACE-1.3B")
            out_data["base_seed"] = payload.get("base_seed", 2025)
            out_data["size"] = payload.get("size", "480p")
            out_data["frame_num"] = payload.get("frame_num", 81)
            out_data["sample_shift"] = payload.get("sample_shift", 16)
            out_data["sample_solver"] = payload.get("sample_solver", "unipc")
            out_data["sample_steps"] = payload.get("sample_steps", 50)
            out_data["sample_guide_scale"] = payload.get("sample_guide_scale", 5.0)
            out_data["save_file"] = payload.get("save_file", None)

            return [out_data]

        # t2v inference
        out_data = payload
        out_data["model_name"] = os.getenv("MODEL_NAME", "Wan2.1-VACE-1.3B")
        out_data["src_video"] = None
        out_data["src_mask"] = None
        out_data["src_ref_images"] = None
        out_data["base_seed"] = payload.get("base_seed", 2025)
        out_data["frame_num"] = payload.get("frame_num", 81)
        out_data["size"] = payload.get("size", "480p")
        out_data["sample_shift"] = payload.get("sample_shift", 16)
        out_data["sample_solver"] = payload.get("sample_solver", "unipc")
        out_data["sample_steps"] = payload.get("sample_steps", 50)
        out_data["sample_guide_scale"] = payload.get("sample_guide_scale", 5.0)
        out_data["save_file"] = payload.get("save_file", None)
        out_data["task"] = payload.get("task", None)

        return [out_data]

    def inference(self, inputs):

        input = inputs[0]

        SIZE_MAP = {
            "480p": (480, 832),
            "720p": (720, 1280),
        }

        src_video_path = input.get("src_video")

        if src_video_path is None:
            logging.info("No source video provided, using default resolution")
            width, height = SIZE_MAP.get(input.get("size", "480p"), SIZE_MAP["480p"])
        else:
            rdr = imageio.get_reader(src_video_path)
            first_frame = rdr.get_next_data()
            rdr.close()
            width, height = first_frame.shape[1], first_frame.shape[0]
            logging.info(f"Source video resolution: {width}x{height}")

        src_video_path = input.get("src_video")
        src_mask_path = input.get("src_mask")
        src_ref_images = input.get("src_ref_images")
        prompt = input.get("prompt", "")
        negative_prompt = input.get("negative_prompt", "")
        num_steps = int(input.get("sample_steps", 30))
        guidance = float(input.get("sample_guide_scale", 5.0))
        base_seed = int(input.get("base_seed", 2025))
        explicit_frames = int(input.get("frame_num", 81))

        # Load refs first (can be used with or without video/mask)
        ref_images = self._load_refs(src_ref_images)

        video_frames = None
        mask_frames = None
        if src_video_path:
            video_frames = self._load_video_frames(src_video_path)
        if src_mask_path:
            mask_frames = self._load_mask_frames(
                src_mask_path, n_hint=(len(video_frames) if video_frames else explicit_frames)
            )

        # conditioning strength — single float is fine for 1 stream
        conditioning_scale = float(input.get("conditioning_scale", 1.0))

        # If no video given, we still can do R2V with refs; use the requested frame count.
        num_frames = len(video_frames) if video_frames is not None else explicit_frames

        g = torch.Generator(device=self.pipe.device).manual_seed(base_seed)

        # floor to avoid Diffusers silently rounding
        # this ensures same dimensions as outputed by VACE
        MULT = 16
        width = (int(width) // MULT) * MULT
        height = (int(height) // MULT) * MULT

        logging.info(f"Using resolution: {width}x{height}")

        logging.info(
            f"Source mask resolution: {(mask_frames[0].width, mask_frames[0].height) if mask_frames is not None else 'N/A'}"
        )

        out_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video=video_frames,
            mask=mask_frames,
            reference_images=ref_images,
            conditioning_scale=conditioning_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=g,
        ).frames[0]

        # Save next to your temp_dir for postprocess()
        save_dir = self.temp_dir
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "out_video.mp4")
        export_to_video(out_frames, out_path, fps=16, quality=8)

        return out_path

    def postprocess(self, inference_output):
        # Read the video file and return its contents
        with open(inference_output, "rb") as vf:
            video_bytes = vf.read()

        # Clean up the file after reading
        try:
            os.remove(inference_output)
        except Exception as e:
            # Print caught exception to if not able to remove the video file
            logging.warning(
                f"Failed to remove temporary file \
                             {inference_output}: {e}"
            )

        return [video_bytes]

    def _load_video_frames(self, path: str):
        rdr = imageio.get_reader(path)
        frames = [Image.fromarray(f).convert("RGB") for f in rdr]
        rdr.close()
        return frames

    def _load_mask_frames(self, path: str, n_hint: int | None):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".webm", ".avi", ".mov", ".mkv"):
            rdr = imageio.get_reader(path)
            frames = [Image.fromarray(f).convert("L") for f in rdr]
            rdr.close()
        else:
            m = Image.open(path).convert("L")
            frames = [m] * (n_hint if n_hint is not None else 1)
        return frames

    def _load_refs(self, csv: str | None):
        if not csv:
            return None
        paths = [p.strip() for p in csv.split(",") if p.strip()]
        return [Image.open(p).convert("RGB") for p in paths]
