## 如何三角化
$s_{uv}=kd_{uv}+b_s+n_s$
$m_{uv}=d_{uv}+n_m$

## 一些问题
1. 在融合的时候应该注意, single view depth 只是视差, 因此是反过来的. 所以可以变化一下, 分别测试使用逆深度和 原始深度进行滤波的效果.
2. 注意融合的时候可以优化一下, single view depth对应的比例和偏置.

## depth weight
local_weight 1 2 3 4, 四个filter都有对应的sigma需要调整. 暂时先这样.

