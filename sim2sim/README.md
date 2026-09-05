# 带可视化,键盘控制  6、7控制转向  箭头键控制前进后退
python sim2sim\sim2sim.py --task 策略 --endpoint_trail --endpoint_bodies 

# 足端轨迹
## 单腿只要 RR 的轨迹
python sim2sim\sim2sim.py --task single_leg_rr_trace --endpoint_trail --endpoint_bodies RR_2

## 四腿参考轨迹(和你截图对应,4 条椭圆)
python sim2sim\sim2sim.py --task quad_leg_track --endpoint_trail --endpoint_trail_points 1000 

## quad_leg_trot1
python sim2sim\sim2sim.py --task quad_leg_trot1 --endpoint_trail 

# quad_leg_free_gait3
python sim2sim\sim2sim.py --task quad_leg_free_gait3 --endpoint_trail --endpoint_bodies --endpoint_trail_points 1000