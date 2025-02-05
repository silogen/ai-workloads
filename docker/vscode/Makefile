APPNAME=openvscode-server
VERSION=vllm-dev-20250124
DOCKERFILE=openvscode-server-vllm-dev.Dockerfile

build:
	docker buildx build -t $(APPNAME):$(VERSION) -f $(DOCKERFILE) .

tag: build
	docker tag $(APPNAME):$(VERSION) ghcr.io/silogen/$(APPNAME):$(VERSION)

push: tag
	docker push ghcr.io/silogen/$(APPNAME):$(VERSION)

clean:
	-docker container rm $(APPNAME)
	docker image rm $(APPNAME)