py -3.12 -m venv venv
sudo bash ./venv/Scripts/activate
source venv/bin/activate
py -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
py -m pip install -U -q huggingface_hub
hf download Salesforce/codet5p-770m --local-dir ./content/codet5p-770m

py -m pip install -q nbmake
py -m pip install -q pytest pytest-timeout

py -m pytest --nbmake "src/notebooks/codet5p-770m.ipynb"
py -m pytest --nbmake "src/notebooks/codet5p-770m-fine-tuned.ipynb"
