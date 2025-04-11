# Logistics image with separate root target

The purpose of the logistics image is to manage data movement from/to various data sources, e.g. Huggingface registry, Minio tenants etc. It contains all the necessary libraries and clients required for downloading and uploading data from/to supported storage providers.

For the interaction between the logistics and main workload containers, both containers usually need access to files created by the other one. This requires that the containers run under the same user. Megatron training images currently require a root user, so we must use an accompanying logistics container which is similarly setup for the root user. To build the logistics image with the root user based target include `--target base` in the `docker build` command.

```bash
DOCKER_BUILDKIT=1 \
docker build -f docker/logistics/logistics.Dockerfile \
             --target base \
             -t ghcr.io/silogen/logistics:v0.1r .
```

If the intent is to build the image with the user having uid `1000`, just skip the `--target base` option.
