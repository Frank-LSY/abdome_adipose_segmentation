from model import UNet
from utils.dataset import ISBI_Loader
from utils.split_data import split_data
from torch import optim
import torch.nn as nn
import torch
import json

N_CLASSES = 3


def train_net(net, device, data_path, epochs=10, batch_size=11, lr=0.00001):
    # 加载训练集
    print('加载数据集')
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    loss_list = []
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            # print(label.shape)
            # print(pred.shape)
            # 将label的灰度维去掉，使可以用交叉熵loss
            label_t = torch.squeeze(label, dim=1).long()
            # print(label_t.shape)

            # print(label_t.shape)
            loss = criterion(pred, label_t)
            print('Loss/train', loss.item())
            loss_list.append(loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'result/best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
#     json.dumps(loss_list)
    file_path = 'result/loss.json'  # 要保存的 JSON 文件路径
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(loss_list))
    
    
if __name__ == "__main__":
    # 拆分训练集，测试集
    print('拆分train test 数据集。。。')
    input_images = '../../data/ori/images'
    input_masks = '../../data/ori/masks'
    train_images = '../../data/train/images'
    train_masks = '../../data/train/masks'
    test_images = '../../data/test/images'
    split_data(input_images=input_images, input_masks=input_masks,
               train_images=train_images, train_masks=train_masks, test_images=test_images)
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为2。
    net = UNet(n_channels=1, n_classes=N_CLASSES)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "../../data/train"
    train_net(net, device, data_path)
