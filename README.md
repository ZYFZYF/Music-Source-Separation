# Music-Source-Separation
    《数字媒体(2)：多媒体》课程中音频小课堂大作业-音源分离任务，实现的算法基于[MUSIC SOURCE SEPARATION USING STACKED HOURGLASS NETWORKS](https://arxiv.org/pdf/1805.08559)
## 运行说明
 - 下载[MIR-1K数据集](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) 到Dataset/目录下(注意不是MIR-1K for MIREX)，确保路径Dataset/MIR-1K/Wavfile存在（其他子文件夹可以不要，注意不要多一层MIR-1K）
 - 运行时先执行
    ~~~
   pip install -r requirements.txt
    ~~~
   来安装程序运行所需要的额外的库，可能在此之外还需要安装ffmpeg，安装方法依系统而不同，此处不再赘述
 - 想要对单个音频文件进行音源分离，运行命令如下
    ~~~
   python main.py --generate src.wav --model model.pt
    ~~~ 
   其中src.wav是指定的要分离的音频的路径，而model.pt是训练好的模型的路径。
 - 想要验证某个模型在测试集上的指标，运行命令如下
    ~~~
   python main.py --test model.pt
    ~~~
## 参考代码
 - https://github.com/sungheonpark/music_source_sepearation_SH_net
 - https://github.com/princeton-vl/pose-ae-train