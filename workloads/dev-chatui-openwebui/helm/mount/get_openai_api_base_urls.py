import os
import sys

import requests
from kubernetes import client, config


def endpoint_check(endpoint):
    try:
        response = requests.get(endpoint.rstrip("/") + "/models")
        if response.status_code == 200:
            return True
    except requests.RequestException:
        pass
    return False


def get_services(separator=";"):
    configuration = config.load_incluster_config()
    v1 = client.CoreV1Api(client.ApiClient(configuration))
    namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
    services = v1.list_namespaced_service(namespace)
    filtered_services = [
        f"http://{svc.metadata.name}/v1"
        for svc in services.items
        if svc.metadata.name.startwith("llm-inference")
    ]
    filtered_services = [
        url
        for url in set(filtered_services).union(
            os.environ.get("OPENAI_API_BASE_URLS", "").split(separator)
            + os.environ.get("OPENAI_API_BASE_URLS_K8S", "").split(separator)
        )
        if endpoint_check(url)
    ]
    env_var_value = separator.join(sorted(filtered_services))
    return env_var_value


sys.stderr.write("Discovered OpenAI API Base URLs\n")
sys.stderr.write(get_services(separator="\n"))
sys.stderr.flush()
print(get_services(separator=";"))
