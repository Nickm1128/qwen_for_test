# Qwen API Server

This repository provides a simple API to interact with the Qwen model. The server exposes a `/chat` endpoint that accepts a list of chat messages and returns the generated response.

## Installation

```bash
pip install -r requirements.txt
```

## Running the server

```bash
python server.py
```

The model weights will be downloaded on the first run.

## Example request

```python
import requests

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
resp = requests.post("http://localhost:8000/chat", json={"messages": messages})
print(resp.json())
```
