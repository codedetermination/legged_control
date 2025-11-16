#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from collections import deque

from std_msgs.msg import Float64MultiArray
from policy_network import Policy

class RLServerNode:
    def __init__(self):
        self.device = torch.device("cpu")

        self.state_dim = 450
        self.action_dim = 12
        self.history_len = 10    # 10 帧 * 45 = 450

        self.policy = Policy(self.state_dim, self.action_dim).to(self.device)

        ckpt_path = rospy.get_param("~ckpt_path")
        rospy.loginfo("Loading RL policy from %s", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.eval()

        # 存最近 10 帧的 45 维 obs
        self.obs_history = deque(maxlen=self.history_len)

        self.obs_sub = rospy.Subscriber("rl_obs", Float64MultiArray,
                                        self.obs_callback, queue_size=1)
        self.tau_pub = rospy.Publisher("rl_tau", Float64MultiArray, queue_size=1)

    def obs_callback(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.shape[0] != 45:
            rospy.logerr_throttle(1.0, "rl_obs length %d != 45", data.shape[0])
            return

        # 1. 更新历史
        self.obs_history.append(data)

        # 不足 10 帧时用第一帧填充
        if len(self.obs_history) == 0:
            return

        while len(self.obs_history) < self.history_len:
            self.obs_history.appendleft(self.obs_history[0].copy())

        # 2. 拼成 450 维 state
        obs_list = list(self.obs_history)  # len = 10, each (45,)
        state_vec = np.concatenate(obs_list, axis=0)  # (450,)
        state_t = torch.from_numpy(state_vec[None, :]).to(self.device)  # [1,450]

        # 3. 调用策略网络：输出 mean，当成 torque
        with torch.no_grad():
            mean, log_std, std = self.policy(state_t)
            action = mean.squeeze(0).cpu().numpy()  # (12,)

        # 4. 发布 12 维 torque
        out = Float64MultiArray()
        out.data = action.astype(np.float64).tolist()
        self.tau_pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("rl_server_node")
    node = RLServerNode()
    rospy.loginfo("RL server node started (topic mode: 45 in, 12 out).")
    rospy.spin()

