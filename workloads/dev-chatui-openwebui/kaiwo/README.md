# Simple Deployment of Open WebUI Without Authentication for Testing LLMs

## Deployment

This works with kaiwo v.0.0.5 only.

```bash
kaiwo serve -p kaiwo  # GPU not required.
```

## Port Forwarding

Forward port 8080 for local access:

```bash
kubectl port-forward deployments/open-webui-main 8080:8080 -n kaiwo    
```

Access the WebUI locally at [http://localhost:8080/](http://localhost:8080/)
