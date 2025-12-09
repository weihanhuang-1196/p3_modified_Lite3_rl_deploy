## 本软件包实现的是一个硬件接口



### 功能如下
+ 接收机器人发布的话题rl_sar/Robot_State，并拆解imu信息和电机状态信息
+ 接收rl_sar发布的话题robot_joint_controller/command
+ 将imu信息发布到/imu话题中，为rl_sar提供imu
+ 根据robot_control.yaml中的update_rate频率自动调用write和read函数
+ write函数将接收到的command命令发送到机器人命令话题rl_sar/Robot_Commands中  
+ read函数将机器人的各电机状态（位置、速度、力矩）发送给control



