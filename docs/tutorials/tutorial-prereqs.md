# Tutorial 0: Prerequisites for running the tutorials

### Required program installs

Programs that are used in this tutorial:

* [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)

* [helm](https://helm.sh/docs/intro/install/)

* [k9s](https://k9scli.io/topics/install/)

* [jq](https://jqlang.org/download/)

* [curl](https://everything.curl.dev/install/index.html)

At least curl and often jq too are commonly installed in many distributions out of the box.

### Additional cluster setup
In order to run the workloads you need to have a Kubernetes cluster with sufficient resources configured. This includes storage, secrets, namespace and HuggingFace token. These should be installed and configured as part of the installation process, but if this is not the case you can use the following command to set up these. The command does the following:

* Adds a namespace, where we will conduct all our work. We will use the `silo` namespace, but please change this to the default namespace in your cluster.

* Adds an External Secret to get the credentials to access the MinIO storage from our namespace.

    - This depends on a ClusterSecretStore called `k8s-secret-store` being already setup by a cluster admin, and the MinIO API credentials being secret there.
    The cluster should have these by default.

* Adds a LocalQueue so that our Jobs schedule intelligently.

    - This references the ClusterQueue `kaiwo` which should already be setup by a cluster admin.

We will use the helm chart in `workloads/k8s-namespace-setup/helm` and the overrides in `workloads/k8s-namespace-setup/helm/overrides/`.

```bash
kubectl create namespace "silo"
helm template workloads/k8s-namespace-setup/helm \
  --values workloads/k8s-namespace-setup/helm/overrides/tutorial-01-local-queue.yaml \
  --values workloads/k8s-namespace-setup/helm/overrides/tutorial-01-storage-access-external-secret.yaml \
  --namespace "silo" \
  | kubectl apply -f - --namespace silo
```

- HuggingFace token: In addition to running the command above you also need to add your HF Token as a secret called `hf-token` with the key `hf-token` in your namespace.