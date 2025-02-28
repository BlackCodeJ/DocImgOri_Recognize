import subprocess
from paddlex import create_model
import re
from sklearn.metrics import accuracy_score
import numpy as np


class Console:
    def __init__(self):
        self.predict_model = "./output/best_model/inference"
        self.input_dir = "./dataset/test/images"

    def predict(self, confidence):
        model = create_model("PP-LCNet_x1_0_doc_ori", model_dir=self.predict_model)
        output = model.predict(self.input_dir, batch_size=1)
        arr = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        # m = {"0": 0, "90": 1, "180": 2, "270": 3}
        m = {"0": 0, "90": 1, "270": 2}
        real_labels = []
        predict_labels = []
        scores = []
        for res in output:
            score = res.json['res']['scores'][0]
            input_path = res.json['res']['input_path']
            real_label = re.search("img_rot([0-9]*)", input_path).group(1)
            real_label = m[real_label]
            predict_label = res.json['res']['label_names'][0]
            predict_label = m[predict_label]
            # 置信度辅助判断
            if score < confidence:
                predict_label = 0
            # 数组操作
            arr[real_label][predict_label] += 1
            # 输出预测失败的结果
            if real_label != predict_label:
                res.print()
            else:
                scores.append(score)

            real_labels.append(real_label)
            predict_labels.append(predict_label)

        print(
            f"预测正确集：[置信度最小值: {min(scores)}, 置信度最大值: {max(scores)}, 置信度中位数: {scores[int(len(scores) / 2)]}, 置信度平均值: {sum(scores) / len(scores)}]")

        print(f"""
        {"实际/预测":<10} {0:<10} {90:<10} {270:<10}
        {0:<10} {arr[0][0]:<10} {arr[0][1]:<10} {arr[0][2]:<10}
        {90:<10} {arr[1][0]:<10} {arr[1][1]:<10} {arr[1][2]:<10}
        {270:<10} {arr[2][0]:<10} {arr[2][1]:<10} {arr[2][2]:<10}
        """)

        real_labels = np.array(real_labels)
        predict_labels = np.array(predict_labels)
        accuracy_all = accuracy_score(real_labels, predict_labels)
        print(f"全样本的准确率: {accuracy_all}")

        # 过滤出标签为1和2的样本
        mask = (real_labels == 1) | (real_labels == 2)
        # 计算过滤后的样本的准确率
        accuracy_mask = accuracy_score(real_labels[mask], predict_labels[mask])
        print(f"翻转样本的准确率: {accuracy_mask}")

        return accuracy_all

# 寻找最佳置信度
def find_best_cf(obj):
    cf = 0.75
    accuracy_max = 0
    cf_best = 0
    while cf < 0.85:
        accuracy = obj.predict(cf)
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            cf_best = cf
        cf += 0.01

    print(f"best_confidence={cf_best}, accuracy={accuracy_max}")


if __name__ == "__main__":
    console = Console()
    # console.predict(0.7)
    find_best_cf(console)
