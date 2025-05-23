# Data_Prep_Agent
This project preprocess an uploaded csv data file. Many kaggle challenges involve simple data preprocessing to enable further machine learning processing. This agent streamlines the process of such preprocessing and codes in a Jupyter Notebook which the user can use directly to add more machine learning frameworks.

# The NVIDIA AIQ Toolkit
This project is built on the [NVIDIA AIQ Toolkit](https://github.com/NVIDIA/AIQToolkit). An environment for the toolkit must be setup and the project folder `data-prep-agent` must be inside the AIQ Toolkit.

Alternatively, you can run the following command
```
git clone https://github.com/NVIDIA/AIQToolkit
cd aiqtoolkit
git submodule update --init --recursive
git lfs install
git lfs fetch
git lfs pull
uv venv --seed .venv
source .venv/bin/activate
uv sync --all-groups --all-extras
aiq --help
aiq --version
export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>
```
and put `data-prep-agent` into the `aiqtoolkit` folder and run `uv pip install -e data_prep_agent`

# Running the project
First start a aiq server through `aiq serve --config_file=data_prep_agent/configs/config.yml`. Then, run the `data_prep_agent/src/data_prep_agent/frontend.py` file and open, if hosting locally, `http://127.0.0.1:5000/`. Upload a single data file, and then ask the agent to perform various data preprocessing and testing on the data. A jupyter notebook can be found at `data_prep_agent/src/jupyter_notebooks` which contains the code that preprocesses the code. You can export the notebook to your colab and use the data to train your machine learning model.