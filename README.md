# VGG-based Telegram Bot for Neural Style Transfer

Simple Aiogram-based Telegram Bot with VGG style transfer model and Redis for running tasks in background.

### Installation

For Docker usage simply run

```
docker-compose build
docker-compose up
```

For non-Docker execution you should follow these steps:

1. Install necessary packages
```
pip install -r requirements.txt
pip install torch torchvision
```

2. Install Redis on your machine and run it on `localhost:6379`
3. Run `bot.py` and `worker.py`

### Examples

| Content | Style | Result |
|---------|-------|--------|
| <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/photos/dicaprio.jpeg?raw=true" width="128" height="128"> | <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/styles/vangogh.jpeg?raw=true" width="128" height="128">   |   |
| <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/photos/dancing.jpeg?raw=true" width="128" height="128"> | <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/styles/vangogh.jpeg?raw=true" width="128" height="128">   |   |
| <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/photos/dicaprio.jpeg?raw=true" width="128" height="128"> | <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/styles/picasso.jpeg?raw=true" width="128" height="128">   |   |
| <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/photos/dancing.jpeg?raw=true" width="128" height="128"> | <img src="https://github.com/noname19871/tg_style_transfer/blob/main/examples/styles/picasso.jpeg?raw=true" width="128" height="128">   |   |

### Inference time

Approximately 2 minutes on Macbook Pro 2019 and ~7 minutes on AWS EC2 t2.small instance


