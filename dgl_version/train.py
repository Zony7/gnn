import torch
import torch.nn.functional as F
from dgl_version import dataloader4kg
from dgl_version.KGCN import train, KGCN, predict

if __name__ == '__main__':
    n_neighbors = 10
    n_layers = 2 # 设置消息传递次数

    users, items, train_set, test_set = dataloader4kg.readRecData(dataloader4kg.Ml_100K.RATING)
    entitys, relations, kgTriples = dataloader4kg.readKgData(dataloader4kg.Ml_100K.KG)
    adj_kg = dataloader4kg.construct_kg(kgTriples)
    adj_entity, adj_relation = dataloader4kg.construct_adj(n_neighbors, adj_kg, len(entitys))
    if len(train_set) == 0:
        raise ValueError("训练数据集为空，请检查数据加载部分")

    # 训练模型
    train( epochs = 10, batchSize = 64, lr = 0.01,

           n_users = max( users ) + 1, n_entitys = max( entitys ) + 1,
           n_relations = max( relations ) + 1, adj_entity = adj_entity,
           adj_relation = adj_relation, train_set = train_set,
           test_set = test_set, n_neighbors = n_neighbors,
           aggregator_method = 'sum', act_method = F.relu, drop_rate = 0.5, n_layers=n_layers )


    # 加载训练好的模型
    model = KGCN( n_users = max( users ) + 1, n_entitys = max( entitys ) + 1,
                  n_relations = max( relations ) + 1, e_dim = 10,
                  adj_entity = adj_entity, adj_relation = adj_relation,
                  n_neighbors = n_neighbors, aggregator_method = 'sum',
                  act_method = F.relu, drop_rate = 0.5, n_layers=n_layers )
    model.load_state_dict(torch.load('model/KGCN_model.pth'))
    model.eval()

    # 示例预测
    user_ids = [119, 172, 445, 13, 6, 6]  # 示例用户ID
    item_ids = [327, 445, 456, 331, 7, 9]  # 示例物品ID
    predictions = predict(model, user_ids, item_ids)
    print("Predictions:", predictions)