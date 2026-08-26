

# Start the Ollama server in its own process group so we can kill it + all its children
setsid bash run_ollama_to_serve_llm > ollama_serve.log 2>&1 &
OLLAMA_PGID=$!

# Guarantee cleanup on ANY exit (normal, error, or Ctrl-C)
cleanup() {
    echo ">>> Shutting down Ollama server (PGID $OLLAMA_PGID)..."
    kill -- -"$OLLAMA_PGID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until the server is actually accepting requests before hitting it
echo ">>> Waiting for Ollama to come up..."
until curl -sf http://127.0.0.1:11435/api/tags >/dev/null 2>&1; do
    sleep 2
done
echo ">>> Ollama is ready."

# Run the job — when this finishes (or fails), the trap fires and kills Ollama
eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate sci-zsel
python alias_generation.py

