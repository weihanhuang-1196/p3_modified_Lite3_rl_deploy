# panda3 仿真以及实机运行流程

> 如需修改或替换模型，请修改[policy/panda3](policy/panda3)中的文件


## xbox、beitong和Sony手柄按钮控制区别

### xbox
- 模型切换
    - 运动学站立 ： B
    - 行走  ：     X
    - 趴下  ：     A
        - 指令
            - 起来：上按钮
    - 低身行走 ：   Y
        - 指令
            - 伏身 ： 右遥感往下
    - 切换到阻尼状态：LB + Y
- 上下使能 ： start


### beitong
- 模型切换
    - 运动学站立 ： B
    - 行走  ：     X
    - 趴下  ：     A
        - 指令
            - 起来：上按钮
    - 低身行走 ：   Y
        - 指令
            - 伏身 ： 右遥感往下
    - 切换到阻尼状态：LB + Y
- 上下使能 ： start



### Sony
- 模型切换
    - 运动学站立 ： ○
    - 行走  ：     △
    - 趴下  ：     ×
        - 指令
            - 起来：上按钮
    - 低身行走 ：   □
        - 指令
            - 伏身 ： 右遥感往下
    - 切换到阻尼状态：L1 + △
- 上下使能 ： R2






## 仿真


### ros2 gazebo
启动gazebo仿真环境
```
./run_gazebo_control.sh
```
启动rl控制器
```
./run_rl.sh
```


### mujoco仿真
启动mujoco仿真环境和rl控制器
```
./run_mujoco.sh
```



## 实机

启动硬件接口
```
./run_real_control.sh
```
启动rl控制器
```
./run_rl.sh
```