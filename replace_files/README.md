## 一点说明
把对应的代码文件放到 T2M-GPT repo 的对应位置。另外，关于 `eval_trans.py`，我把采样次数从 30 降到 4，比如 `evaluation_transformer_test, line 356`，加速评测进度，担心跑不完。如果时间充裕，可以自行修改。

`eval_trans.py` -> `T2M-GPT/utils/eval_trans.py`

`t2m_trans.py` -> `T2M-GPT/models/t2m_trans.py`

`GPT_eval_multi_modified.py` -> `T2M-GPT/*`