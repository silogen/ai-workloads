import argparse
import os

import requests  # type: ignore
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString


# This script uses ruamel.yaml to preserve comments and formatting in YAML files.
def get_pipeline_tag(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("pipeline_tag")


def get_chat_template_flag(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    config = data.get("config", {})
    tokenizer_config = config.get("tokenizer_config", {})
    chat_template = tokenizer_config.get("chat_template")
    chat_template_jinja = config.get("chat_template_jinja")
    # Check for config.processor_config.chat_template
    processor_config = config.get("processor_config", {})
    processor_chat_template = processor_config.get("chat_template")
    return chat_template is not None or chat_template_jinja is not None or processor_chat_template is not None


def update_yaml_file(filepath, dry_run=False):
    yaml_inst = YAML()
    yaml_inst.preserve_quotes = True
    yaml_inst.indent(mapping=2, sequence=4, offset=2)
    with open(filepath, "r") as f:
        data = yaml_inst.load(f)
    # Try both 'model' and 'modelID' keys
    model_id = data.get("model") or data.get("modelID")
    if not model_id:
        return False
    pipeline_tag = get_pipeline_tag(model_id)
    chat = get_chat_template_flag(model_id)
    # Ensure metadata and metadata.labels exist
    metadata = data.get("metadata")
    if metadata is None:
        data["metadata"] = metadata = yaml_inst.map()
    labels = metadata.get("labels")
    if labels is None:
        metadata["labels"] = labels = yaml_inst.map()
    # Ensure quoted string values for YAML output
    labels["pipeline_tag"] = DoubleQuotedScalarString(str(pipeline_tag) if pipeline_tag is not None else "")
    labels["chat"] = DoubleQuotedScalarString("true" if chat else "false")
    # Remove deprecated fields if present
    if "pipeline_tag" in metadata:
        del metadata["pipeline_tag"]
    if "chat" in metadata:
        del metadata["chat"]
    if dry_run:
        print(
            f"[DRY RUN] Would update labels.pipeline_tag: \"{labels['pipeline_tag']}\", labels.chat: \"{labels['chat']}\" in {filepath}"
        )
    else:
        with open(filepath, "w") as f:
            yaml_inst.dump(data, f)
        print(
            f"Updated labels.pipeline_tag: \"{labels['pipeline_tag']}\", labels.chat: \"{labels['chat']}\" in {filepath}"
        )
    return True


def main():
    parser = argparse.ArgumentParser(description="Update YAML files with HuggingFace pipeline_tag.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, but do not modify files.")
    args = parser.parse_args()
    yaml_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                filepath = os.path.join(root, file)
                yaml_files.append(filepath)
    for filepath in sorted(yaml_files):
        try:
            update_yaml_file(filepath, dry_run=args.dry_run)
        except Exception as e:
            print(f"Failed to update {filepath}: {e}")


if __name__ == "__main__":
    main()
