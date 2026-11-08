"""Advance reward curriculum after successful PPO updates and checkpoint it."""
from mjlab.rl.runner import MjlabOnPolicyRunner


class RowingRunner(MjlabOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original_update=self.alg.update
        def update():
            result=original_update()
            self.env.unwrapped.rowing.penalty_curriculum.advance()
            return result
        self.alg.update=update

    def save(self,path,infos=None):
        info=dict(infos or {})
        info['rowing_penalty_curriculum']=self.env.unwrapped.rowing.penalty_curriculum.state_dict()
        super().save(path,info)

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        info=super().load(path,load_cfg,strict,map_location)
        curriculum=self.env.unwrapped.rowing.penalty_curriculum
        if load_cfg is None or load_cfg.get('iteration',True):
            if info and 'rowing_penalty_curriculum' in info:
                curriculum.load_state_dict(info['rowing_penalty_curriculum'])
            else:
                # Upstream checkpoint iter is the last completed zero-based update.
                curriculum.set_iteration(self.current_learning_iteration+1)
            # learn() starts at this index; do not replay the saved update index.
            self.current_learning_iteration=curriculum.completed_iterations
        else:curriculum.set_iteration(0)
        return info
