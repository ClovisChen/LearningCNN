1. FrameArray 是一个dict用于保存当前sliding window中的Frame, 但是现在窗口中的frame也就是关键帧的选择的策略是有问题的.
2. Gaussian Newton的结果到底应该怎样验证?
    - 正向验证
    - 要把residual打印出来看看
    - 图像warp之后的结果?
    - 反向验证雅克比矩阵
3. 现在还没有做好motion estimation.
4. current level is good.
5. level变化的问题, 代码实现的时候,对于帧内先遍历层,然后是帧, 然后遍历点. 
    - tracker->layers->frames->points
    - 
## motion model
1. constant translation or orientation.
2. LK estimate the motion.


## KeyFrame Select
1. traced points less than threshold.
2. 


## scipy vs g2o
scipy 相对更加灵活, 工作方式更加原始, 可以手动设计迭代中所使用的数据.
g2o的话则更加成熟, 使用方式也比较固定. 


|item | scipy | g2o |
|---|---|---|
|优点 | 灵活 | 成熟|
|使用方式 | func, data | edge, vertex|
| 稀疏性设置 | 通过矩阵设置 | 自动完成|


## scipy vs pygtsam
github来源了gtsam的python版(pygtsam),因此可以考虑使用pygtsam对后端进行优化.

## about IMU
- 如何初始化?
- 后端的优化

