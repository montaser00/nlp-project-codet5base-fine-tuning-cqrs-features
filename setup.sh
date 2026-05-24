python -m venv venv
sudo bash ./venv/Scripts/activate
python -m pip install -r requirements.txt

python -m pip install -U -q huggingface_hub
hf download Salesforce/codet5p-770m --local-dir ./content/codet5p-770m

python -m pip install -q nbmake
python -m pip install -q pytest pytest-timeout

python -m pytest --nbmake "src/notebooks/codet5p-770m.ipynb" "src/notebooks/codet5p-770m-fine-tuned.ipynb" --timeout=900
