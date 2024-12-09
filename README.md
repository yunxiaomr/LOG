# LOG
This repository contains code for the paper "LOG: A Local-to-Global Optimization Approach for Retrieval-based Explainable Multi-Hop Question Answering". 

To run our code, you will first need to download the dataset files in the raw format. 
Note that officially released data and what we have used here are only different in the format (e.g. uses different names for json fields), and are not qualitatively different. 
Take a look at raw_data_to_official_format.py if you're interested. 

## Environments
python 3.8. 

We suggest you to create a virtual environment with: conda create -n LOG python=3.8.19

Then activate the environment with: conda activate LOG

Install packages: pip install -r requirements.txt

## Running example
### Retrieval Model
```
python run.py train experiment_configs/select_and_answer_model_selector_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_selector_for_musique_ans
            	       python run.py predict serialization_dir/select_and_answer_model_selector_for_musique_ans/model.tar.gz \
                      raw_data/musique_ans_dev.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent
```

### Reader Model
```
python run.py train experiment_configs/select_and_answer_model_answerer_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_answerer_for_musique_ans
            	      python run.py predict serialization_dir/select_and_answer_model_answerer_for_musique_ans/model.tar.gz \
                      serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_dev.jsonl \
                      --output-file serialization_dir/select_and_answer_model_answerer_for_musique_ans/predictions/serialization_dir__select_and_answer_model_selector_for_musique_ans__predictions__musique_ans_dev.jsonl \
                      --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent
```

## Result  
You will get the result files at last. If you want to convert predictions to the official format, run:
            	python raw_predictions_to_official_format.py serialization_dir/select_and_answer_model_answerer_for_musique_ans/predictions/serialization_dir__select_and_answer_model_selector_for_musique_ans__predictions__musique_ans_dev.jsonl

You can use evaluate_v1.0.py to evaluate your predictions against ground-truths. For eg.:
            	python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_end2end_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl

## Dependencies
- torch==1.7.1
- allennlp==2.1.0
- numpy==1.23.0
- pandas==2.0.3
- scikit_learn==1.3.2
- protobuf==5.27.1
- transformers==4.7.0

## Citation
Please kindly cite our paper if the work is helpful.
```bibtex
@inproceedings{xu-etal-2025-log,
    title = "{LOG}: A Local-to-Global Optimization Approach for Retrieval-based Explainable Multi-Hop Question Answering",
    author = "Xu, Hao  and
      Zhao, Yunxiao and
      Zhang, Jiayang and 
      Wang, Zhiqiang  and
      Li, Ru",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)",
    year = "2025"
}
```

## Acknowledgement

The code is largely based on [MusiQue](https://github.com/stonybrooknlp/musique). We appreciate their contributions to the community.

