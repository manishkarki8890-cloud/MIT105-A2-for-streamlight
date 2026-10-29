# BaadhiSuchak (Streamlit) - next-day flood risk for Nepal river stations

Deploy on Streamlit Community Cloud: upload ALL files and the model/ folder to the top level of a
GitHub repo, then create an app with Main file path = streamlit_app.py (Python 3.12 in Advanced settings).

Run locally:  pip install -r requirements.txt  &&  streamlit run streamlit_app.py

If the host installs a different scikit-learn version than the one the model was saved with,
the app automatically retrains the same Random Forest on start-up (about 10-30 s, once).
