## Training

下载 Weizmann Horse 数据集（https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata），并置于 Datasets 文件夹中，按照如下格式：

![image-20220818163454760](C:\Users\xmttttt\AppData\Roaming\Typora\typora-user-images\image-20220818163454760.png)

(由于代码中使用 torchvision.datasets.ImageFolder 进行数据集导入，故需要两级目录)

然后直接运行 train.py 即可

```
python train.py
```

训练后将会在 weights 文件夹中每 10 个 epochs 保存一组权重，并在主目录下生成 result.jpg，保存训练时重要参数的走向。

<img src=".\result.jpg" alt="result" style="zoom:80%;" />

## Model

训练过程中使用的模型文件（包括主干网 ResNet50_FCN 与 ASPP Head）都存于 model.py 文件中。

训练使用的损失函数为 BCELoss，存于 loss_func.py 中（该源文件是在作 L2 Loss 尝试时使用）



## Evaluation

测试时使用的文件为 eval.py，包括 IOU 与 Boundary IOU 的计算函数和使用测试集进行测试的主函数。

测试前需要将已训练好的权重文件 best.pth 置于 weights 文件夹下。

链接：https://pan.baidu.com/s/1pRxJnb-cHEJuuvacXDHl1A 
提取码：552l

然后直接运行 eval.py 即可

```
python eval.py
```