# EvenAI Gemini Bridge

OpenAI-compatible API bridge to Google Gemini.

## Setup

```bash
uv sync
cp config.example.yaml config.yaml
```

## Configuration

Edit `config.yaml` with your settings:

- `gemini_api_key` (required) - Your Google Gemini API key
- `token` (required) - Token to authorize requests with
- `gemini_model` (optional) - Model to use (default: `gemini-3-flash-preview`)
- `server.port` (optional) - Server port (default: `8000`)

Edit `system_prompt.md` to customize the system prompt sent to Gemini.

## Run

```bash
uv run python main.py
```

## Configure in app
In the Even app go to Settings -> Even AI -> Agent Configure. From there add a new agent.  
Name it as you like, for the URL use `http://<your_host>/v1/chat/completions` 
and for the token use the same one that you put into the config.yaml.

## Docker

You can also run the application using Docker:

1. Build the image:
```bash
docker build -t evenai-gemini-bridge .
```

2. Run the container:
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/system_prompt.md:/app/system_prompt.md \
  evenai-gemini-bridge
```

Make sure you have created your `config.yaml` file locally before running the container.


Server runs on `http://localhost:8000`

## API

- `POST /v1/chat/completions` - OpenAI-compatible chat endpoint
- `GET /health` - Health check
