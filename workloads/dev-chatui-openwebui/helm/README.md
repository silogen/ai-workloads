# Open WebUI for LLM Services

This Helm Chart deploys a WebUI for aggregating deployed LLM services within the cluster.

## Deploying the Workload

Basic configurations are defined in the `values.yaml` file.

To deploy the service, run the following command within the Helm folder:

```bash
helm template <release-name> . | kubectl apply -f -
```

### Automatic Discovery of LLM services

#### Server-side discovery

When role-binding is set up properly, the deployment will perform server-side automatic discovery of LLM inference services within the namespace (i.e. service names that start with "llm-inference"). The deployment will verify all endpoint URLs and include them in the list only if a request to the `/models` path returns a 200 status code.

#### Client-side discovery

Additionally, client-side automatic discovery can be enabled by setting the `--dry-run=server` flag. You can specify any OpenAI-compatible endpoints by setting the `env_vars.OPENAI_API_BASE_URLS` environment variable.

To enable client-side automatic discovery, use the following command for example:

```bash
helm template <release-name> . \
    --set env_vars.OPENAI_API_BASE_URLS="http://example-open-ai-api-server.com/v1/" \
    --dry-run=server |\
    kubectl apply -f -
```
