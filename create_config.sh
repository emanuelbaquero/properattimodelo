mkdir -p ~/.streamlit
echo "[server]
headless = true 
port = $PORT 
enableCORS = False
" > ~/.streamlit/config.toml

