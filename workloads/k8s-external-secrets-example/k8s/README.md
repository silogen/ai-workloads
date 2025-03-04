# ExternalSecret: hf-token

This example demonstrates how to use the External Secrets Operator to synchronize secrets from an external secret management system into a Kubernetes cluster. Specifically, it shows how to configure an `ExternalSecret` resource to fetch a secret from a pre-configured external secret store and create a corresponding Kubernetes secret. This setup is particularly useful for managing sensitive information securely within Kubernetes environments.


## Description


The `ExternalSecret` resource defined in this example fetches a secret from an external secret store and creates a corresponding Kubernetes secret in the `kaiwo` namespace.


## Setup and Check Prerequisites


Before using this `ExternalSecret`, ensure the following prerequisites are met:


### Cluster Secret Store

A `ClusterSecretStore` named `gcp-secret-store` must be set up. This store provides access to the external secret management system where the secrets are stored. To verify that it exists in the cluster, use the following `kubectl` command:

```bash
kubectl get clustersecretstore
```


### External Secret

A secret called `hf-token` must exist in the external secret store. This secret will be synchronized into the Kubernetes cluster as a Kubernetes secret.
