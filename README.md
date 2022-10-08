# Learning Behavior-Aware Cognitive Diagnosis for Online Education Systems

Source code and data set for the paper *Learning Behavior-Aware Cognitive Diagnosis for Online Education Systems*.

The code is the implementation of LKT model, and the data set is the public data set [ASSIST2009-2010] and MOOC-LKT (https://sites.google.com/site/assistmentsdata/home/assistment2009-2010-data/skill-builder-data-2009-2010
).

If this code helps with your studies, please kindly cite the following publication:

```
@inproceedings{mao2021learning,
  title={Learning Behavior-Aware Cognitive Diagnosis for Online Education Systems},
  author={Mao, Yiming and Xu, Bin and Yu, Jifan and Fang, Yifan and Yuan, Jie and Li, Juanzi and Hou, Lei},
  booktitle={International Conference of Pioneering Computer Scientists, Engineers and Educators},
  pages={385--398},
  year={2021},
  organization={Springer}
}
```



## Dependencies:

- python 3.6
- pytorch >= 0.10 (pytorch 0.4 might be OK but pytorch<0.4 is not applicable)
- numpy
- json
- sklearn



## Usage

Run divide_data.py to divide the original data set data/log_data.json into train set, validation set and test set. The data/ folder has already contained divided data so this step can be skipped.

`python divide_data.py`

Train the model:

`python train.py {device} {epoch}`

For example:

`python train.py cuda:0 5`  or `python train.py cpu 5`

Test the trained the model on the test set:

`python predict.py {epoch}`



## Data Set

The data/log_data.json is extracted from public data set [ASSIST2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment2009-2010-data/skill-builder-data-2009-2010) (skill-builder, corrected version) where `answer_type != 'open_response' and skill_id != ''`. And the logs are organized in the structure:

- log_data.json = [user1, user2, ...]
- user = {"user_id": user_id, "log_num": log_num, "logs": [log1, log2, ...]]}
- log = {"exer_id": exer_id, "score": score, "knowledge_code": [knowledge_code1, knowledge_code1, ...]}



The user_id, exer_id and knowledge_code correspond to user_id, problem_id and skill_id attributes in the original ASSIST2009-2010 csv file. The ids/codes are recoded starting from 1.



## Details

- Students with less than 15 logs would be deleted in divide_data.py.
- The model parameters are initialized with Xavier initialization.
- The model uses Adam Optimizer, and the learning rate is set to 0.002.
