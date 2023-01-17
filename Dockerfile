FROM python:3.10

ENV BOT_API_TOKEN=YOUR_BOT_TOKEN_HERE

WORKDIR /app/tg_style_transfer
ADD ./requirements.txt /app/tg_style_transfer/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .


EXPOSE 8000
RUN chmod +x start.sh
CMD ["./start.sh"]
