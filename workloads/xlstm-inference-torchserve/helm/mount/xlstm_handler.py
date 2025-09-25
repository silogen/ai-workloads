import json
import logging
import os
import zipfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class xLSTMHandler(BaseHandler):
    """
    TorchServe handler using HuggingFace Transformers for causal language modeling.
    """

    def __init__(self):
        super(xLSTMHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """
        Initialize the handler, loading model and tokenizer.

        Args:
            ctx: Context object with system properties and manifest.

        The method:
        1. Sets up the device (GPU or CPU) based on system properties
        2. Extracts the model.zip file from the model directory
        3. Loads the tokenizer and model using HuggingFace's transformers
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
            zip_ref.extractall(model_dir)

        weights_dir = os.path.join(model_dir, "model")
        tokenizer_dir = os.path.join(model_dir, "tokenizer")

        # Load tokenizer and model
        logger.info(f"Loading tokenizer from {tokenizer_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.info(f"Tokenizer loaded successfully from {tokenizer_dir}")

        logger.info(f"Loading model from {weights_dir}")
        self.model = AutoModelForCausalLM.from_pretrained(weights_dir)
        logger.info(f"Model loaded successfully from {weights_dir}")

        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, requests):
        """
        Extract a prompt string from each incoming request and
        tokenize it into input_ids for the model.
        Supports:
        - POST /predictions with JSON under "body", "data", or "prompt"
        - POST /invocations with a raw string
        """
        inputs = []
        for idx, data in enumerate(requests):
            logger.info(f"[preprocess] request #{idx}: {data!r}")
            prompt = None

            # 1) If they sent a dict with "body" or "data"
            if isinstance(data, dict):
                raw = data.get("body") or data.get("data")
                if isinstance(raw, dict):
                    prompt = raw.get("prompt")
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                if isinstance(raw, str):
                    # try JSON parse
                    try:
                        obj = json.loads(raw)
                        prompt = obj.get("prompt")
                    except json.JSONDecodeError:
                        prompt = raw
                # fallback to top‐level "prompt" key
                if prompt is None:
                    prompt = data.get("prompt")

            # 2) If they sent a raw string (invocations endpoint)
            elif isinstance(data, str):
                prompt = data

            # 3) If they sent raw bytes
            elif isinstance(data, (bytes, bytearray)):
                prompt = data.decode("utf-8")

            if not isinstance(prompt, str):
                logger.error("No prompt found in request #%d: %r", idx, data)

            inputs.append(prompt)

        if not inputs:
            raise ValueError("No valid prompts provided")

        # tokenize the batch of prompts
        encoding = self.tokenizer(
            inputs,
            return_tensors="pt",
        )

        return encoding["input_ids"].to(self.model.device)

    def inference(self, input_ids):
        """
        Generate text.
        """
        # Generate up to 1000 new tokens as per the example
        outputs = self.model.generate(input_ids, max_new_tokens=1000, do_sample=True)
        return outputs

    def postprocess(self, inference_output):
        """
        Decode generated token IDs to text outputs.
        """
        results = []
        decoded = self.tokenizer.batch_decode(inference_output, skip_special_tokens=True)
        for decoded_output in decoded:
            results.append({"output": decoded_output})
        return results
