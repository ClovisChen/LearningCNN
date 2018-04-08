# learningReloc
1. Trainging a net used for relocalization, different from posenet, this net use features maps to match point correspondence in current frame and maps.


在测试数据的时候,分别测试了kitti raw data, hobot road data, hobot garage data.

## test monodepth net
在运行代码之前,首先生成一个filelist, filelist指定了要test的文件列表,可以使用代码prepare_data.py 中的make_file_name_list函数执行.如果原始文件为bag数据,可以使用load_ros_bag.py中的save_bag_images函数保存bag中的message至image文件,这里把双目的左右图像分别保存至left,right文件夹.

### test kitti
进入代码的根目录(${LEARNINGRELOC})
```
python ${LEARNINGRELOC}net/utils/depth_utils.py

```


### test hobot garage/road data
```
python ${LEARNINGRELOC}net/utils/depth_utils_hobot.py
```


## add FCN segment  

