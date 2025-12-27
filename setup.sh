#!/bin/bash
# setup.sh - For Streamlit Cloud deployment

# Install backend dependencies
pip install -r requirements.txt

# Start backend server in background
nohup uvicorn backend:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start
sleep 3

# Start Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0