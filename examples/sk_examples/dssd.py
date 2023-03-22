import pandas as pd

from tods import schemas as schemas_utils,load_pipeline
from tods import generate_dataset, evaluate_pipeline

table_path = 'metric_detection/staircase_data/staircase_data_0_from2018-12-19to2019-01-31_8305.csv'
target_index = 2 # what column is the target
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
df["label"] = df["label"].astype(int)
# print(df)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
# pipeline = schemas_utils.load_default_pipeline()
pipeline=load_pipeline("abc.json")
# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)