"""Completed PPO-update curriculum, independent of batch size or episode resets."""
import torch


class PenaltyCurriculum:
    def __init__(self,num_envs,device,threshold=1000):
        self.threshold=threshold
        self.completed_iterations=0
        self.action_scale=torch.ones(num_envs,device=device)
        self.angular_scale=torch.ones(num_envs,device=device)

    def set_iteration(self,count):
        if int(count)!=count or count<0:raise ValueError('Invalid iteration count')
        self.completed_iterations=int(count)
        late=count>=self.threshold
        self.action_scale.fill_(40. if late else 1.) # -0.05 -> -2
        self.angular_scale.fill_(5. if late else 1.) # -1 -> -5

    def advance(self):
        self.set_iteration(self.completed_iterations+1)

    def scale(self,kind):
        if kind=='action':return self.action_scale
        if kind=='angular':return self.angular_scale
        raise ValueError('Unknown curriculum penalty')

    def state_dict(self):
        return {'completed_iterations':self.completed_iterations,'threshold':self.threshold}

    def load_state_dict(self,state):
        self.set_iteration(state['completed_iterations'])
