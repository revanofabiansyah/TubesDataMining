# TubesDataMining

## How to run the web app

1. Create environment and activate

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r model/requirements.txt
```

3. Run the web model
```bash
python model/app.py
```

4. Open http://localhost:5000 in your browser

5. Use the web app to make predictions (the feedback require Ollama with `gemma3:1b-it-fp16` model)

## How to install Ollama

> [!IMPORTANT]
> 
> TO run Gemma3:1b-it-fp16 model, you need GPU with at least 3GB of VRAM for it to run smoothly. Otherwise, the Ollama will use CPU (much slower).

1. Download Ollama from https://ollama.com/download
2. Follow the instruction / run the installer

### For windows (App)

1. Run the ollama program (for windows)
2. Wait for Ollama to start
3. Click the `Select Model` button (in Windows app) then find and select `gemma3:1b-it-fp16`
4. Wait for the model to be downloaded
5. To check if the model is running, try send a message (in the app)

### CLI (for Linux)

1. Run the following command in terminal to start Ollama
```bash
ollama serve
```
2. Wait for Ollama to start
3. On a different terminal, run the following command in terminal to pull the model
```bash
ollama pull gemma3:1b-it-fp16
```
4. Wait for the model to be downloaded
5. To test the model, first run the following command in terminal to start the model
```bash
ollama run gemma3:1b-it-fp16
```
6. Wait for the model to be ready
7. Send a message to the model to check if it's running
