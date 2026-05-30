py -3.12 -m venv venv
sudo bash ./venv/Scripts/activate
py -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
py -3.12 -m pip install -U -q huggingface_hub
hf download Salesforce/codet5p-770m --local-dir ./content/codet5p-770m

py -3.12 ./src/features/evaluate/codet5p-770m-evaluate.py
