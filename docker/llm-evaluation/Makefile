APPNAME=evaluation-workloads-metrics
VERSION=v0.1
DOCKERFILE=Dockerfile

build:
	docker buildx build --platform linux/amd64 -t $(APPNAME):$(VERSION) .

tag: build
	docker tag $(APPNAME):$(VERSION) ghcr.io/silogen/$(APPNAME):$(VERSION)

push: tag
	docker push ghcr.io/silogen/$(APPNAME):$(VERSION)

clean:
	-docker container rm $(APPNAME):$(VERSION)
	docker image rm $(APPNAME):$(VERSION)
