
import matplotlib.pyplot as plt
import numpy as np


class DeltaPID(object):
    """增量式PID算法实现"""

    def __init__(self, target, cur_val, dt, p, i, d) -> None:
        self.dt = dt  # 循环时间间隔
        self.k_p = p  # 比例系数
        self.k_i = i  # 积分系数
        self.k_d = d  # 微分系数

        self.target = target  # 目标值
        self.cur_val = cur_val  # 算法当前PID位置值
        self._pre_error = 0  # t-1 时刻误差值
        self._pre_pre_error = 0  # t-2 时刻误差值

    def calcalate(self):
        error = self.target - self.cur_val
        p_change = self.k_p * (error - self._pre_error)
        i_change = self.k_i * error
        d_change = self.k_d * (error - 2 * self._pre_error + self._pre_pre_error)
        delta_output = p_change + i_change + d_change  # 本次增量
        self.cur_val += delta_output  # 计算当前位置

        self._pre_pre_error = self._pre_error

        self._pre_error = error

        return self.cur_val


    def fit_and_plot(self, count=200):
        counts = np.arange(count)
        outputs = []
        for i in counts:
            outputs.append(self.calcalate())
            print('Count %3d: output: %f' % (i, outputs[-1]))

        print('Done')

        plt.figure()
        plt.axhline(self.target, c='red')
        plt.plot(counts, np.array(outputs), 'b.')
        plt.ylim(min(outputs) - 0.1 * min(outputs),
                max(outputs) + 0.1 * max(outputs))
        plt.plot(outputs)
        plt.show()


if __name__ == "__main__":
    pid = DeltaPID(100, -80, 0.5, 0.2, 0.1, 0.001)
    pid.fit_and_plot(150)