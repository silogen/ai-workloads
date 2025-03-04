#!/bin/bash

# 0. ASSUMPTION: workload has been deployed
# 1. ASSUMPTION: port-forward is set up, e.g. (assuming deployment with this name)
#
#       kubectl port-forward deployments/ubuntu-llama-3-1-8b-instruct-hf  8080:8080 -n kaiwo
#
# 2. run this script using
#
#       ./test.sh
#
#    or
#       ./test.sh | jq

curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
