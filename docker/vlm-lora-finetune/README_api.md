# FastAPI OpenCLIP LORA train/inference endpoint (WIP)

## API usage

Check `api_main.py` for endpoint code.

- `/train`: takes in a YAML string (clipora config) and a ZIP file containing training data. This will start a training run in a separate process. It will save checkpoints of LORA layers in `TRAIN_JOB_OUTPUT_DIR`. The job will update a sqlite database file while training. Returns the job id which can be used to check status of job. Training data must have training and evaluation CSV files and the training images with this kind of structure:

```
train.csv
eval.csv
path/to/image1.png
path/to/image2.png
...
```

Where CSVs look like:

```
"image_path","language_instruction"
"path/to/image1.png","put the cube on top of the cylinder"
"path/to/image2.png","Move the blue spoon to the left burner"
...
```

NOTE: when using API the image paths in the CSVs have to be relative to the ZIP folder since it may get extracted to an arbitrary folder!

- `/status/{job_id}`: returns status information about a job.
- `/classes_inference`: runs inference using a fine-tuned model on a single image. Requires the job id, an image and a list of classes. OpenCLIP doesn't return text, it gives the text and images in an embedding space. This example inference gives the probabilities for the list of classes given.

## Example run

You can run these commands to give the API a try. Run in this folder:

```
# start a barebones Deployment and Service that git pulls dev-fastapi branch, installs reqs and starts uvicorn:
kubectl create -f testing_deployment.yaml
# when running, port forward to local on port 8080:
kubectl port-forward services/clipora-testing-deployment 8080:80
# use small test files to test things out. this should return successfully and give a job id
curl -X POST "http://localhost:8080/train/" \\n  -F "config_str=<tests/fixtures/bridge_small/api_example_config.yml" \\n  -F "file=@tests/fixtures/bridge_small/bridge_dataset_small.zip;type=application/zip"
# get the job id from last command and check for status:
curl "http://localhost:8080/status/{job_id}"
# when status is "complete", try running a single inference with the fine-tuned model (uses latest checkpoint):
# Note: may take half a minute
curl -X POST "http://localhost:8080/inference/?job_id=db65b5dd-508e-403c-bcee-2116ac27e205" \
-F "image=@/Users/tman/data/bridge_dataset_small/episode_0001/step_0014.png" \
-F "classes=Move the food item to the lower left side of the table
Slide the green rag in front of the sushi.
put pear in bowl"
# Return value should have probabilities and classes
# Download the fine-tuned lora layers and config as zip:
curl http://localhost:8080/download_finetuned_model/db65b5dd-508e-403c-bcee-2116ac27e205 --output downloaded_model.zip
# You can also upload a ZIP containing clipora config and lora layers, let's try with the downloaded zip:
curl -X POST "http://localhost:8080/upload_finetuned_lora/" \
-F "file=@/Users/tman/work/downloaded_model.zip;type=application/zip"
# the command will create a new job where `best_finetuned_lora_path` will point to the uploaded folder
# Clean up if needed:
kubectl delete -f testing_deployment.yaml
```

Note that the results are underwhelming with the small test set and may not even be correct.

By default checkpoints and db file etc. will be written to /tmp/something, check the env vars and defauls in `api_main.py`

## Notes

- Check tests/fixtures/bridge_small/api_example_config.yml for an example of a clipora config.
- unit tests are NOT working yet, you can ignore them!
- Launching the models is kinda slow?
- using Hugging Face datasets not tested.
