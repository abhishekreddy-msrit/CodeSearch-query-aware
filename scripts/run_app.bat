@echo off
cd /d "%~dp0\.."
echo Starting Code Search prototype...
python -m streamlit run app\streamlit_app.py --server.headless true
