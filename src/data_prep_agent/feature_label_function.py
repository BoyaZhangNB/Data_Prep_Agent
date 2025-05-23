import json
import logging
from typing import Any

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
import os

logger = logging.getLogger(__name__)

def write_csv_headers_to_txt():
    """
    Reads the first CSV file in data_prep_agent/src/data_files and writes its headers
    to a .txt file in the same directory, in the format: [col1, col2, col3, ...]
    """
    data_folder = "data_prep_agent/src/data_files"
    data_folder = os.path.abspath(os.path.expanduser(data_folder))
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if not csv_files:
        logger.warning("No CSV files found in the data_files directory.")
        return
    first_csv = csv_files[0]
    csv_path = os.path.join(data_folder, first_csv)
    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    headers = [col.strip() for col in header_line.split(",") if col.strip()]

    return headers


class FeatureLabelIdentifierConfig(FunctionBaseConfig, name="feature_label_identifier"):
    description: str
    llm_name: LLMRef  # Name of the LLM to use
    prompt: str = """
        {input}

        In the context of machine learning, features are the input variables used to make predictions, while labels are the output variables that we want to predict.
        Identify and discern the features and labels in a given list of column hearders below
        Note to exclude not meaningful columns like "id", "timestamp", etc.

        {headers}

        Identify the features and labels in the above list of column headers, and give a response in JSON format.

        example:
            Input: "[pixel_id, band1, band2, band3, color]"
            Output: "features": ["band1", "band2", "band3"], "labels": ["color"]
    """
    _type: str = "feature_label_identifier"

@register_function(config_type=FeatureLabelIdentifierConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def feature_label_identifier(config: FeatureLabelIdentifierConfig, builder: Builder) -> Any:
    """Register the feature label identifier tool."""


    async def _identify_feature_labels(instructions: str) -> str:
        """
        Identify and discern the features and labels in a given list of column hearders.
        In the context of machine learning, features are the input variables used to make predictions, while labels are the output variables that we want to predict.
        Note to exclude not meaningful columns like "id", "timestamp", etc.
        Args:
            instructions: Instructions on how to preprocess the data
        Returns:
            JSON string containing the identified features and labels

        """
        logger.info("Extracting headers from CSV files...")
        headers = write_csv_headers_to_txt()

        logger.info("Identifying features and labels...")
        # Get LLM from builder
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        # Get response from LLM
        response = await llm.apredict(config.prompt.format(headers=headers, input=instructions), temperature=0.1, max_tokens=2000)


        logger.info("LLM response received. Parsing JSON...")

        # Parse the response
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx]

        print("Response:", response)


        # Check that the JSON object has "features" and "labels"
        max_retries = 3
        for attempt in range(max_retries):
            result_json = json.loads(response)
            if isinstance(result_json, dict) and "features" in result_json and "labels" in result_json:
                break
            else:
                logger.warning(f'Attempt {attempt+1}: JSON response missing "features" or "labels": {response}')
                # Regenerate response with LLM
                response = await llm.apredict(config.prompt.format(headers=headers, input=input), temperature=0.5, max_tokens=2000)
            response = await llm.apredict(config.prompt.format(headers=headers, input=input), temperature=0.5, max_tokens=2000)

        data_folder = "data_prep_agent/src/output"
        data_folder = os.path.abspath(os.path.expanduser(data_folder))

        # Save the response as a JSON file in the Data-Prep-GPT directory
        output_dir = os.path.join(os.path.dirname(data_folder), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "feature_label_identifier_result.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)

        return f"Feature and label identification completed. Run the notebook_generation tool to generate the notebook. {instructions}"

    yield FunctionInfo.from_fn(_identify_feature_labels, description=config.description)
