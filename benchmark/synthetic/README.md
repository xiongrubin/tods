# Revisiting Time Series Outlier Detection:Definitions and Benchmarks

This branch is the source code of  experiment part of our paper. We provide everything needed when running the experiments: Dataset, Dataset Generator, Pipeline json, Python script, runner and the result we get from the experiments.

## Resources
* Paper: [Revisiting Time Series Outlier Detection: Definitions and Benchmarks](https://openreview.net/forum?id=r8IvOsnHchr)

## Cite this Work:
If you find this  work useful, you may cite this work:
```
@inproceedings{
    lai2021revisiting,
    title={Revisiting Time Series Outlier Detection: Definitions and Benchmarks},
    author={Kwei-Herng Lai and Daochen Zha and Junjie Xu and Yue Zhao and Guanchu Wang and Xia Hu},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
    year={2021},
    url={https://openreview.net/forum?id=r8IvOsnHchr}
}
```

## Datasets

This All of the datasets were generated from the generators. Tuning parameters in the generators can get your own datasets.

## Pipeline

This Pipeline json files are organized by different settings of algorithms. There are 5 outlier ratios in each setting.

## Runner

To run a pipeline, you can generate your own pipeline json file from script. Take AutoEncoder as an example:

```python
python script/simple_algo/build_AutoEncoder_pipeline.py 
```
Then run the json using run_pipeline in /runner
```python
python runner/run_pipeline.py --pipeline_path ae_pipeline_default_con0.05.json 
```



Or you can also use the pipelines we provided in /Pipeline:

```python
python runner/run_pipeline.py --pipeline_path Pipeline/ae_pipeline_default/ae_pipeline_default_con0.05.json
```



*Please refer master branch of TODS for details of running pipelines.
