source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_free_gait\quad_leg_free_gait3
现在这个训练的，走的很稳、很顺。
接下来几个步骤，一个个完成：
 1、再写一个quad_leg_free_gait4，相较于quad_leg_free_gait3，改动项为，足端Z向高度再大一倍即可，因为以我之前的真机迁移经验，训练时的抬腿高度真机迁移后就不太行了，会拖地行走。  
 2、写一个sim2sim，和之前一样。
 3、写一个真机迁移的，我看看你写的代码我进行真机迁移到底好不好。
 4、我想根据现在训练的quad_leg_free_gait3，出一些数据、图。包括：
前进后退的8个关节电机扭矩、功率等等、左右行走的电机扭矩、功率等等、转弯行走的电机扭矩、功率等等(不确定哪些图值得出，你自己判断下)，以及其他值得出的图。


5、训练完quad_leg_free_gait4，提供：
“我需要你提供的不是数字,是训练后的现象
现在什么也不用给我。等你跑一次 free_gait4 训练,把结果反馈给我,我来判断要不要动这两个数:

训练后看到的现象	我的动作
步态稳、真机能抬腿不再拖地	不用动,这两个就是好值 ✅
走得稳但真机还是拖脚	把 weight 再加大(更狠地罚拖地)或 min_clearance 抬高
步态变得夸张、乱踢/腿抬过头、反而慢	把 weight 减小 或 k 调小
站着不动的环境无缘无故被扣分/站得不安稳	说明门控(moving/has_support)要对,反馈给我
所以流程是:你先跑 rsl_rl_train.py --task Isaac-BennettRL-Flat-QuadLeg-FreeGait4-v0,把 episode_return 曲线和 play 里步态的样子发我 —— 要是觉得还不够好,我再改这两个值,你重训。”



free_gait4训练完了，效果可以说是非常差！！！我又看到了我之前效果（改来改去就是对抬腿高度改不好）：
1、训练到500轮就崩溃了
  File "D:\Conda\envs\env_isaaclab\Lib\site-packages\torch\distributions\normal.py", line 74, in sample
    return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: normal expects all elements of std >= 0.0

2、我在想怎么才能让你看到机器人走的有多差？你看不到视频，你讲讲怎么让你知道机器人走的效果？？