source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_free_gait\quad_leg_free_gait3
现在这个训练的，走的很稳、很顺。
接下来几个步骤，一个个完成：
# step1
 1、再写一个quad_leg_free_gait4，相较于quad_leg_free_gait3，改动项为，足端Z向高度再大一倍即可，因为以我之前的真机迁移经验，训练时的抬腿高度真机迁移后就不太行了，会拖地行走。  

 # step2
 2、写一个sim2sim，和之前一样。

 # step3
 3、写一个真机迁移的，我看看你写的代码我进行真机迁移到底好不好。

 # step4
 4、我想根据现在训练的quad_leg_free_gait3，出一些数据、图。包括：
前进后退的8个关节电机扭矩、功率等等、左右行走的电机扭矩、功率等等、转弯行走的电机扭矩、功率等等(不确定哪些图值得出，你自己判断下)，以及其他值得出的图。


# step5
我让你出这些图的含义：
假设我还没选电机，那么就依据这个数据选电机。
现在我认为只有平地的还不够，那么这个GO2（E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\tasks\manager_based\go2），里边是包含上下楼梯、崎岖平面、上下坡等等，那么数据就多了。
所以咱们也可以写一个bennett_go2,简单就行，不用考虑真机迁移，主要是训练出他的并行策略，然后咱们依据数据出图。
你看没毛病吧。有问题直接问。

# step6


# step7


# step8


# step9


# step10


# step11






# step10
free_gait4训练完了，效果可以说是非常差！！！我又看到了我之前效果（改来改去就是对抬腿高度改不好）：
1、训练到500轮就崩溃了
  File "D:\Conda\envs\env_isaaclab\Lib\site-packages\torch\distributions\normal.py", line 74, in sample
    return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: normal expects all elements of std >= 0.0

2、我在想怎么才能让你看到机器人走的有多差？
logs\rsl_rl\quad_leg_free_gait\quad_leg_free_gait3这个有足端轨迹，logs\rsl_rl\quad_leg_free_gait\quad_leg_free_gait4这个也有，你看看吧。free_gait3相对好些，free_gait4就差很多。