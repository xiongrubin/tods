import pandas as pd
from axolotl.backend.simple import SimpleRunner
from tods import generate_dataset, generate_problem,evaluate_pipeline,load_pipeline
from tods.searcher import BruteForceSearch
import glob
import os

#设置连接多个文件的路径
files = os.path.join("metric_detection/changepoint_data/", "*.csv")

#返回的合并文件列表
files = glob.glob(files)

print("在特定位置加入所有 CSV 文件后生成的 CSV...");

#使用 concat 和 read_csv 加入文件
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df["label"] = df["label"].astype(int)
# df.to_csv("low_signal-to-noise_ratio_data.csv",index_label="timestamp")
# print(df)

# Some information
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_GOOG.csv' # The path of the dataset
target_index = 2 # what column is the target
time_limit = 30 # How many seconds you wanna search

#metric = 'F1' # F1 on label 1
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem

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

print('Pipeline json:', best_pipeline.to_json())



print('*' * 52)
print('Search History:')
for pipeline_result in search.history:
    print('-' * 52)
    print('Pipeline id:', pipeline_result.pipeline.id)
    print(pipeline_result.scores)
print('*' * 52)

print('')

print('*' * 52)
print('Best pipeline:')
print('-' * 52)
print('Pipeline id:', best_pipeline.id)
print('Pipeline json:', best_pipeline.to_json())
print('Output:')
print(best_output)
print('Scores:')
print(best_scores)
print('*' * 52)
with open("abc.json",
          'w') as f:
    f.write(best_pipeline.to_json())
dataset = generate_dataset(df, target_index)

# Load the default pipeline
# pipeline = schemas_utils.load_default_pipeline()
pipeline=load_pipeline("abc.json")
# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)