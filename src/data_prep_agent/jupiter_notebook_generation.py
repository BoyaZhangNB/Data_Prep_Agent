import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

import re

def generate_notebook(input, output_path, dataset_path, label_key, feature_keys):
    # Initialize output list
    cells = []

    cells.append(new_code_cell(
        f"import pandas as pd\n\ndf = pd.read_csv('{dataset_path}') #Replace with true data file path\ndf.head()"
    ))

    # Basic EDA
    cells.append(new_code_cell(
        f"""
# Convert DataFrame to numpy arrays and split into features and labels
X = df[{feature_keys}].values
y = df[{label_key}].values

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
        """
    ))

    # Extract and label
    pattern = re.compile(r'(code block|comment block):\s*\n?')
    matches = list(pattern.finditer(input))
    for i, match in enumerate(matches):
        block_type = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(input)
        content = input[start:end].strip()
        if block_type == "code block":
            cells.append(new_code_cell(content))
        elif block_type == "comment block":
            cells.append(new_markdown_cell(content))


    nb = new_notebook()
    nb['cells'] = cells

    # Write notebook to file
    with open(output_path, "w") as f:
        nbformat.write(nb, f)


import json
import logging
from typing import Any

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

class JupiterNotebookGenerationConfig(FunctionBaseConfig, name="jupiter_notebook_generation"):
    description: str
    llm_name: LLMRef  # Name of the LLM to use
    plan_llm_name: LLMRef  # Name of the LLM to use
    prompt: str = """
        user inquery: {input}

        Generate the code in blocks that the broken down to distinct functionalities. include markdown comment.

        in you response, write in the format similiar to:
        code block: <CODE>
        code block: <CODE>
        comment block: <MARKDOWN>

        do not include any system messages, and do not include code fences

        There is two existing variables in the context:
        X, a numpy array of features
        y, a numpy array of labels
        """

@register_function(config_type=JupiterNotebookGenerationConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def jupiter_notebook_generation(config: JupiterNotebookGenerationConfig, builder: Builder) -> Any:
    """Register the jupiter notebook generation tool."""
    # Parse the input

    logger.info("Generating Jupyter notebook for data preprocessing...")

    async def _generate_jupiter_notebook(instructions: str) -> str:
        """
        Generate a Jupyter notebook for data preprocessing.
        Args:
            instructions: very detailed instructions on how to preprocess the data. Include the machine learning framework to be used and the specific processings to be done.
        Returns:
            Path to the generated Jupyter notebook
        """
            # Load label_key and feature_keys from JSON file
        with open("data_prep_agent/src/output/feature_label_identifier_result.json", "r") as f:
            data = json.load(f)
        label_key = data["labels"]
        feature_keys = data["features"]
        dataset_path = "data_prep_agent/src/data_files/train.csv"
        # Generate the notebook
        output_path = "data_prep_agent/src/jupyter_notebooks/generated_notebook.ipynb"


        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        print(config.prompt.format(input=instructions))

        # Get response from LLM
        response = await llm.apredict(config.prompt.format(input=instructions), temperature=0.1, max_tokens=1000)
        generate_notebook(response, output_path, dataset_path, label_key, feature_keys)

        print(response)

        return "Notebook generated successfully at: " + output_path

    yield FunctionInfo.from_fn(_generate_jupiter_notebook, description=config.description)
