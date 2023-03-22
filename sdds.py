from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
#
# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)
#
# Step 1: column_parser
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: extract_columns_by_semantic_types(attributes)
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)

# Step 3: extract_columns_by_semantic_types(targets)
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_3.add_output('produce')
step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
pipeline_description.add_step(step_3)

attributes = 'steps.2.produce'
targets = 'steps.3.produce'

# Step 4: processing
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
step_4.add_output('produce')
pipeline_description.add_step(step_4)
#
# # Step 5: algorithm`
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6: Predictions
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

# Output to json
data = pipeline_description.to_json()
with open('autoencoder_pipeline.json', 'w') as f:f.write(data)
print(data)


# import pandas as pd
#
# from tods import schemas as schemas_utils
# from tods import generate_dataset, evaluate_pipeline,load_pipeline
#
# table_path = './yahoo_sub_5.csv'
# target_index = 6 # what column is the target
# metric = 'F1_MACRO' # F1 on both label 0 and 1
#
# # Read data and generate dataset
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, target_index)
#
# # Load the default pipeline
# pipeline = schemas_utils.load_default_pipeline()
# # pipeline = load_pipeline('autoencoder_pipeline.json')
# # Run the pipeline
# pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
# print(pipeline_result)

# import sys
# import argparse
# import os
# import pandas as pd
#
# from tods import generate_dataset, load_pipeline, evaluate_pipeline
# table_path = './yahoo_sub_5.csv'
# target_index = 6 # which column is the label
# pipeline_path = "./autoencoder_pipeline.json"
# metric = "ALL"
#
# # Read data and generate dataset
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, target_index)
#
# # Load the default pipeline
# pipeline = load_pipeline(pipeline_path)
#
# # Run the pipeline
# pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
# print(pipeline_result.scores)








# import pandas as pd
#
# from axolotl.backend.simple import SimpleRunner
#
# from tods import generate_dataset, generate_problem
# from tods.searcher import BruteForceSearch
#
# # Some information
# table_path = 'yahoo_sub_5.csv'
# target_index = 6 # what column is the target
# time_limit = 30 # How many seconds you wanna search
# metric = 'F1_MACRO' # F1 on both label 0 and 1
#
# # Read data and generate dataset and problem
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, target_index=target_index)
# problem_description = generate_problem(dataset, metric)
#
# # Start backend
# backend = SimpleRunner(random_seed=0)
#
# # Start search algorithm
# search = BruteForceSearch(problem_description=problem_description,
#                           backend=backend)
#
# # Find the best pipeline
# best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=time_limit)
# best_pipeline = best_runtime.pipeline
# best_output = best_pipeline_result.output
#
# # Evaluate the best pipeline
# best_scores = search.evaluate(best_pipeline).scores










# from d3m import index
# from d3m.metadata.base import ArgumentType
# from d3m.metadata.pipeline import Pipeline, PrimitiveStep
#
# # Creating pipeline
# pipeline_description = Pipeline()
# pipeline_description.add_input(name='inputs')
#
# # Step 0: dataset_to_dataframe
# step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
# step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
# step_0.add_output('produce')
# pipeline_description.add_step(step_0)
#
# # Step 1: column_parser
# step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
# step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
# step_1.add_output('produce')
# pipeline_description.add_step(step_1)
#
# # Step 2: extract_columns_by_semantic_types(attributes)
# step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
# step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
# step_2.add_output('produce')
# step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
#                         data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
# pipeline_description.add_step(step_2)
#
# # Step 3: extract_columns_by_semantic_types(targets)
# step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
# step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
# step_3.add_output('produce')
# step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
#                             data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
# pipeline_description.add_step(step_3)
#
# attributes = 'steps.2.produce'
# targets = 'steps.3.produce'
#
# # Step 4: processing
# step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler'))
# step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
# step_4.add_output('produce')
# pipeline_description.add_step(step_4)
#
# # Step 5: algorithm
# step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
# step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
# step_5.add_output('produce')
# pipeline_description.add_step(step_5)
#
#  # Step 6: Predictions
# step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
# step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
# step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
# step_6.add_output('produce')
# pipeline_description.add_step(step_6)
#
# # Final Output
# pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')
#
# # Output to json
# data = pipeline_description.to_json()
# with open('example_pipeline.json', 'w') as f:
#     f.write(data)
#     print(data)







# import pandas as pd
#
# from tods import schemas as schemas_utils
# from tods import generate_dataset, evaluate_pipeline,load_pipeline
#
# table_path = 'datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv'
# target_index = 6 # what column is the target
# metric = 'F1_MACRO' # F1 on both label 0 and 1
#
# # Read data and generate dataset
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, target_index)
#
# # Load the default pipeline
# # pipeline = schemas_utils.load_default_pipeline()
# pipeline = load_pipeline('autoencoder_pipeline.json')
#
# # Run the pipeline
# pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
# print(pipeline_result)
#
#
#
# import pandas as pd
#
# from axolotl.backend.simple import SimpleRunner
#
# from tods import generate_dataset, generate_problem
# from tods.searcher import BruteForceSearch
#
# # Some information
# table_path = 'datasets/yahoo_sub_5.csv'
# target_index = 6 # what column is the target
# time_limit = 30 # How many seconds you wanna search
# metric = 'F1_MACRO' # F1 on both label 0 and 1
#
# # Read data and generate dataset and problem
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, target_index=target_index)
# problem_description = generate_problem(dataset, metric)
#
# # Start backend
# backend = SimpleRunner(random_seed=0)
#
# # Start search algorithm
# search = BruteForceSearch(problem_description=problem_description,
#                           backend=backend)
#
# # Find the best pipeline
# best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=time_limit)
# best_pipeline = best_runtime.pipeline
# best_output = best_pipeline_result.output
#
# # Evaluate the best pipeline
# best_scores = search.evaluate(best_pipeline).scores



# import pandas as pd
#
# from tods import schemas as schemas_utils
# from tods import generate_dataset, evaluate_pipeline
#
# table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
# target_index = 6 # what column is the target
# metric = 'F1_MACRO' # F1 on both label 0 and 1
#
# # Read data and generate dataset
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, target_index)
#
# # Load the default pipeline
# pipeline = schemas_utils.load_default_pipeline()
#
# # Run the pipeline
# pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
# print(pipeline_result)


import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods import generate_dataset, generate_problem
from tods.searcher import BruteForceSearch

# Some information
table_path = 'yahoo_sub_5.csv'
target_index = 6 # what column is the target
time_limit = 30 # How many seconds you wanna search
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index=target_index)
problem_description = generate_problem(dataset, metric)

# Start backend
backend = SimpleRunner(random_seed=0)

# Start search algorithm
search = BruteForceSearch(problem_description=problem_description,
                          backend=backend)

# Find the best pipeline
best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=time_limit)
best_pipeline = best_runtime.pipeline
best_output = best_pipeline_result.output

# Evaluate the best pipeline
best_scores = search.evaluate(best_pipeline).scores