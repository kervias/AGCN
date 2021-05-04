# Graduation Design Project - AGCN
> Multi-task Algorithm for item recommendation task(IR) and attribute inference task(AI)

## Install
```shell
pip install torch==1.8.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Task
每个任务执行时都会分配一个唯一ID,如 20210420215755 程序运行过程中，结果会保存到 temp文件夹，程序正常结束时，结果会从temp文件夹移动到output文件夹，因此output文件夹中总会保存成功执行的结果

### Item Recommendation Task

#### Train
```shell
python -u main.py --task IR --istrain true --dataset_name movielens1M
```
#### Test
```shell
python -u main.py --task IR --istrain false --dataset_name movielens1M --train_id 20210420215755
```

### Attribute Inference Task
```shell
python -u main.py --task AI --dataset_name movielens1M
```

## Parameters Settings
`/src/conf/datasets/`文件夹下保存了不同数据集的配置文件。 以`amazonVideoGames`为例

```yaml
name: amazonVideoGames
user_count: 31027
item_count: 33899

epoch_num: 300 # epoch times
iter_num: 10 # iter times
free_emb_dim: 32 # free embedding dim
learning_rate: 0.0005 # learning rate
batch_size: 5120
gamma: 0.001
lambda1: 0.001
lambda2: 0.001
attr_union_dim: 32
gcn_layer: 3
neg_item_num: 5
test_topks: [5,10,15,20,25,30,35,40,45,50]

user_attr:
  have: false
item_attr:
  have: true
  attr_type_num: 3
  attr_dim_list: [14, 52, 10]
  attr_type_list: [ 1, 1, 0 ] # 0 means single-label attribute and 1 means multi-label attributes
```

## Data format

