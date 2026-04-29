#!/bin/bash

set -euxo pipefail

# Default flags
skip_pvc=false
skip_configmap=false

# Parse flags
while [[ "$#" -gt 1 ]]; do
  case "$1" in
    --skip-pvc)
      skip_pvc=true
      shift
      ;;
    --skip-configmap)
      skip_configmap=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--skip-configmap] [--skip-pvc] <RayJob name>"
      exit 1
      ;;
  esac
done

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [--skip-configmap] [--skip-pvc] <RayJob name>"
  echo "This script attaches ConfigMap and PVC to the lifecycle of the RayJob for automatic garbage collection."
  echo "ConfigMap name and PVC name must match the RayJob name."
  exit 1
fi

job_name="$1"
job_uid="$(kubectl get rayjob "$job_name" -o=jsonpath='{.metadata.uid}')"

echo "Processing RayJob $job_name (UID: $job_uid)..."

if [ "$skip_configmap" = false ]; then
  echo "Attaching ConfigMap $job_name to the lifecycle of the RayJob."
  kubectl patch configmap "$job_name" --patch $'
metadata:
  ownerReferences:
    - apiVersion: ray.io/v1
      blockOwnerDeletion: true
      controller: true
      kind: RayJob
      name: '$job_name'
      uid: '$job_uid'
'
else
  echo "Skipping ConfigMap patch."
fi

if [ "$skip_pvc" = false ]; then
  echo "Attaching PVC $job_name to the lifecycle of the RayJob."
  kubectl patch pvc "$job_name" --patch $'
metadata:
  ownerReferences:
    - apiVersion: ray.io/v1
      blockOwnerDeletion: true
      controller: true
      kind: RayJob
      name: '$job_name'
      uid: '$job_uid'
'
else
  echo "Skipping PVC patch."
fi

echo "Script completed."
