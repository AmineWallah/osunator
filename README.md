# Osunator
is a deep learning model that generates an osu! replay for a chosen map, based on tensorflow

this repo is the full pipeline of the model

more detailed description to be added soon:tm:

# Docker

[![Docker Hub](https://img.shields.io/docker/v/aminewallah/osunator?logo=docker&label=Docker%20Hub)](https://hub.docker.com/r/aminewallah/osunator)

To run it:
```python
docker run --rm \
  -v "/your/osu/song/folder":/maps \
  -v /tmp/out:/out \
  aminewallah/osunator:latest \
  "/maps/your-map-name.osu" -o /out
```

# Special thanks
To the oomfies who contributed with their replay data in the early versions of the model, THANK YOU: 300mm, JaViLuMa, pluk, MrFish