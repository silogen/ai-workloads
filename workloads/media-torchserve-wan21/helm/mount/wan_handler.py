import json
import logging
import uuid
import zipfile
from abc import ABC

import diffusers
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


class DiffusersHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to video generation.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Wan2.1 model is
            loaded and initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        self.pipe = DiffusionPipeline.from_pretrained(model_dir + "/model")
        self.pipe.to(self.device)
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        inputs = []
        for _, data in enumerate(requests):
            logger.info(f"Full incoming data: {data}")
            input_text = None
            width = int(data.get("width", 480))
            height = int(data.get("height", 832))
            num_frames = int(data.get("num_frames", 81))
            num_inference_steps = int(data.get("num_inference_steps", 40))

            if "body" in data:
                body = data["body"]
                if isinstance(body, (str, bytes, bytearray)):
                    try:
                        if isinstance(body, (bytes, bytearray)):
                            body = body.decode("utf-8")
                        body_json = json.loads(body)
                        input_text = body_json.get("prompt")
                        width = int(body_json.get("width", width))
                        height = int(body_json.get("height", height))
                        num_frames = int(body_json.get("num_frames", num_frames))
                        num_inference_steps = int(body_json.get("num_inference_steps", num_inference_steps))
                    except Exception as e:
                        logger.error(f"Failed to parse body as JSON: {e}")
                elif isinstance(body, dict):
                    input_text = body.get("prompt")
                    width = int(body.get("width", width))
                    height = int(body.get("height", height))
                    num_frames = int(body.get("num_frames", num_frames))
                    num_inference_steps = int(body.get("num_inference_steps", num_inference_steps))
            if input_text is None:
                input_text = data.get("prompt")
            if input_text is None:
                logger.error("No prompt found in request data! Data was: %s", data)
                continue
            logger.info(
                f"Received text: '{input_text}', width: {width}, "
                f"height: {height}, num_frames: {num_frames}, "
                f"num_inference_steps: {num_inference_steps}"
            )
            inputs.append(
                {
                    "prompt": input_text,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                }
            )
        return inputs

    def inference(self, inputs):
        """Generates video relevant to the received text."""
        video_paths = []
        for inp in inputs:
            output = self.pipe(
                prompt=inp["prompt"],
                negative_prompt=None,
                height=inp["height"],
                width=inp["width"],
                num_frames=inp["num_frames"],
                guidance_scale=5.0,
                num_inference_steps=inp["num_inference_steps"],
            ).frames[0]

            video_path = f"output_{uuid.uuid4()}.mp4"
            export_to_video(output, video_path, fps=16)
            logger.info(f"Generated video: {video_path}")
            video_paths.append(video_path)
        return video_paths

    def postprocess(self, inference_output):
        """Post Process Function to handle video output.
        Args:
            inference_output (list): It contains paths to the generated video files.
        Returns:
            list: Returns a list of binary video data or URLs.
        """
        video_data = []
        for video_path in inference_output:
            with open(video_path, "rb") as vid_file:
                video_data.append(vid_file.read())
        return video_data
