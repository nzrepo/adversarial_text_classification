# adversarial-text-classification
## 背景
将CV领域的对抗训练思想，迁移应用到NLP中。
NLP领域的基于对抗样本的对抗训练，可用于帮助模型提高泛化性，起到提升性能的目的。可以参见[1]。

## 实验过程
### 数据准备&backbone模型框架
以https://github.com/649453932/Chinese-Text-Classification-Pytorch 项目为实验主体，选取char-TextCNN模型作为主要训练框架，最大限度复用了其代码。
数据组成：从THUCNews中抽取的20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。
类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。
数据集划分：
|  数据集   | 数据量  |
|  ----  | ----  |
| 训练集  | 18w |
| 验证集  | 1w |
| 测试机  | 1w |

### 实验参数
baseline通用参数：  
batch_size = 64  
embedding_size = 300  
seq_length = 32  
early-stop机制：验证集loss超过1000个batch没下降，结束训练  
|  方法   | 特殊参数  |
|  ----  | ----  |
| baseline  | / |
| FGSM  | epsilon=0.02 |
| PGD  | K=3, epsilon=0.02 |
| Free  | K=5, epsilon=0.02 |

### 实验结果
|  方法   | PRECISION  |  RECALL  |  F1-SCORE  | 
|  ----  | ----  |  ----  |  ----  |
| baseline  | 90.41% |  90.33%  |  90.34%  |
| FGSM  | 90.64% |  90.66%  |  90.64%  |
| PGD  | 91.01% |  91.01%  |  91%  |
| Free  | 88.07% |  88.06%  |  88.03%  |  

 

各方法的准确率收敛曲线：  
![image](https://user-images.githubusercontent.com/102469274/160290049-ab221359-974a-4fc4-8162-e0c81bbc7619.png)


## 运行
python run_adv.py 

### requirements
<li>pytorch  
<li>tensorboardX  
<li>sklearn  

## 参考文献
[1] [炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)  
[2] [locuslab/fast_adversarial](https://github.com/locuslab/fast_adversarial)  
[3] [649453932/Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)  
[4] [enricivi/adversarial_training_methods](https://github.com/enricivi/adversarial_training_methods)  
[5] [【综述】NLP 对抗训练（FGM、PGD、FreeAT、YOPO、FreeLB、SMART）](https://blog.csdn.net/qq_38204302/article/details/120774826?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_aa&utm_relevant_index=2)  
[6] [对抗训练的理解，以及FGM、PGD和FreeLB的详细介绍](https://blog.csdn.net/weixin_41712499/article/details/110878322)  
[7] [FreeLB: Enhanced Adversarial Training for Language Understanding](https://arxiv.org/abs/1909.11764v5)  
[8] [FGSM: Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572v1.pdf)  
[9] [FGM: Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/pdf/1605.07725.pdf)  

