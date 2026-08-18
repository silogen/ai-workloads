#!/usr/bin/env python
"""
Transforms a kubernetes manifest file by wrapping all Deployments, Jobs etc. in the manifest
in their equivalent Kaiwo CRD. For example Deployment -> KaiwoService

Input is taken from stdin. Output is stdout. Output is identical to input except Deployments
etc. are wrapped in their Kaiwo equivalents

Usage:

helm template example . | python mutate_manifest.py | kubectl apply -f -
or
cat rendered_manifest.yaml | python mutate_manifest.py > transformed_manifest.yaml

Requires ruamel.yaml==0.18.10 package. May not work with older ruamel.yaml versions
"""

import copy
import sys

from ruamel.yaml import YAML

kaiwo_type_mapping = {
    "Deployment": {"kind": "KaiwoService", "spec_key": "deployment"},
    "RayService": {"kind": "KaiwoService", "spec_key": "rayService"},
    "Job": {"kind": "KaiwoJob", "spec_key": "job"},
    "RayJob": {"kind": "KaiwoJob", "spec_key": "rayJob"},
}


def wrap_with_kaiwo(doc):
    """Wraps a kubernetes resource such as Deployment in its Kaiwo equivalent

    Args:
        doc (dict): The kubernetes resource to wrap

    Returns:
        dict: The resource wrapped with Kaiwo
    """
    # Copy labels from original to kaiwo and add 'app' with same name as the original
    labels = copy.deepcopy(doc["metadata"].get("labels", {}))
    name = doc["metadata"]["name"]

    labels["app"] = name
    kaiwo_values = kaiwo_type_mapping[doc["kind"]]
    new_metadata = {"name": name}
    new_metadata["labels"] = labels

    return {
        "apiVersion": "kaiwo.silogen.ai/v1alpha1",
        "kind": kaiwo_values["kind"],
        "metadata": new_metadata,
        "spec": {kaiwo_values["spec_key"]: doc},
    }


def wrap_deployments(stream):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Filter out empty manifests
    docs = [doc for doc in yaml.load_all(stream) if doc is not None]
    new_docs = []

    for doc in docs:
        # If not a yaml or it doesn't have Kaiwo CRD equivalent, leave unchanged
        if not isinstance(doc, dict) or doc.get("kind") not in kaiwo_type_mapping:
            new_docs.append(doc)
            continue

        kaiwo_crd = wrap_with_kaiwo(doc)
        new_docs.append(kaiwo_crd)

    yaml.dump_all(new_docs, sys.stdout)


if __name__ == "__main__":
    wrap_deployments(sys.stdin)
