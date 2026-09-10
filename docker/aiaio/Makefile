APPNAME=aiaio
VERSION=v$(shell date +%Y%m%d)
DOCKERFILE=Dockerfile

checkout:
	-git clone https://github.com/abhishekkrthakur/aiaio.git
	cd aiaio
	git pull

build: checkout
	cd aiaio &&	docker buildx build -t $(APPNAME):$(VERSION) -f $(DOCKERFILE) .

tag: build
	docker tag $(APPNAME):$(VERSION) ghcr.io/silogen/$(APPNAME):$(VERSION)

push: tag
	docker push ghcr.io/silogen/$(APPNAME):$(VERSION)

clean:
	-docker container rm $(APPNAME)
	docker image rm $(APPNAME)
