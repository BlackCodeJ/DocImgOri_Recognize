"# DocImgOri_Recognize_model"
0. 本项目在通用图像方向分类模型上对文档图像方向分类效果进行增强，本模型在自定义数据集上测试准确率为97.5%，若增加置信度考量准确率可以高达98.96%

1. 环境搭建：
   - python<=3.10 
   - paddleclas                2.6.0 
   - paddlepaddle-gpu          3.0.0b1 
   - paddlex                   3.0.0b2

2. 下载paddlex仓库，我这里用的版本是release/3.0-rc
    仓库地址：<https://github.com/PaddlePaddle/PaddleX>

3. 下载paddleclas仓库，我这里用的版本是release/2.6
    将paddleclas文件夹放到该路径中:paddlex/paddlex/repo_manager/repos

4. 将本仓库中的文件放到下载的paddlex仓库下  
项目路径： 
```
paddlex:  
    |-- model # 模型权重参数等文件  
    |-- test.py # 测试模型效果
```

注意事项(报错)：
1. 报错：`paddlex.utils.errors.others.UnsupportedPa ramError: 'PP-LCNet_x1_0_doc_ori' is not a registered model name.`
   - 可能原因：项目对应插件没装
   - 解决方法：仔细对照上述1,2条进行操作，参考文献：https://blog.csdn.net/2301_80977231/article/details/144314736

2. 报错：`NotADirectoryError: [WinError 267] 目录名称无效`
   - 可能原因：paddleclas没有放到正确目录下
   - 解决方法：参照上述第三条

3. 报错：  
(1)`TypeError: expected str, bytes or os.PathLike object, not NoneType.`  
(2)`TypeError: paddlex.inference.models.image_classification.ClasPredictor() got multiple values for keyword argument 'model_dir'`
   - 可能原因：代码中'from paddlex import create_model'导入出现问题。应该导入paddlex-release/3.0-rc版本的paddlex包，而实际导入的是前版本的包
   - 解决方法：  
   (1)在项目根目录下导入最新版paddlex包，包下载地址：https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-rc/paddlex (推荐)  
   (2)将create_model的参数修改为self.predict_model