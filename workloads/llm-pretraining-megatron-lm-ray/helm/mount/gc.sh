#!/bin/bash

set -euxo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <RayJob name>"
  echo "This script attaches configmap and pvc to the lifecycle of the RayJob for automatic garbage collection."
  echo "Config map name and pvc name must be the same as the name of the RayJob."
  exit 1
fi

job_name="$1"

job_uid="$(kubectl get rayjob $job_name -o=jsonpath='{.metadata.uid}')"

echo "Attaching ConfigMap $job_name and PVC $job_name to the lifecycle of the RayJob $job_name."

kubectl patch configmap $job_name --patch $'
metadata:
  ownerReferences:
    - apiVersion: ray.io/v1
      blockOwnerDeletion: true
      controller: true
      kind: RayJob
      name: '$job_name'
      uid: '$job_uid'
'

kubectl patch pvc $job_name --patch $'
metadata:
  ownerReferences:
    - apiVersion: ray.io/v1
      blockOwnerDeletion: true
      controller: true
      kind: RayJob
      name: '$job_name'
      uid: '$job_uid'
'

echo "Patched ConfigMap $job_name and PVC $job_name successfully."
