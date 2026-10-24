"""The first-stage task, constructed without inherited locomotion rewards."""
from mjlab.envs import ManagerBasedRlEnvCfg, mdp as common
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.utils.noise import UniformNoiseCfg
from .asset import entity_cfg
from . import mdp


def env_cfg(num_envs=2048,play=False):
    return ManagerBasedRlEnvCfg(
        decimation=10, episode_length_s=30.0,
        scene=SceneCfg(num_envs=num_envs,env_spacing=0.0,entities={'rowing':entity_cfg()},
            sensors=(ContactSensorCfg(name='forbidden_contact',
                primary=ContactMatch(mode='geom',pattern='h2_collision_.*',entity='rowing'),
                secondary=ContactMatch(mode='geom',pattern='(h2|oar)_collision_.*',entity='rowing'),
                fields=('found',),reduce='netforce',num_slots=1),)),
        # Retain the repository 2ms timestep for stiff welds; control is 50Hz.
        sim=SimulationCfg(njmax=512,nconmax=128,contact_sensor_maxmatch=256,
                          mujoco=MujocoCfg(timestep=0.002,integrator='implicitfast',iterations=50)),
        actions={'h2':mdp.RowingActionCfg(entity_name='rowing')},
        observations={
            'actor':ObservationGroupCfg(terms={
                'proprio':ObservationTermCfg(func=mdp.onboard_proprio,
                    noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01)),
                'action':ObservationTermCfg(func=mdp.last_action),
                'command':ObservationTermCfg(func=mdp.onboard_command),
            },history_length=122,flatten_history_dim=False,enable_corruption=not play),
            'privileged':ObservationGroupCfg(terms={'state':ObservationTermCfg(func=mdp.privileged)},
                                             enable_corruption=False),
        },
        events={
            'reset_scene':EventTermCfg(func=common.reset_scene_to_default,mode='reset'),
            'hydrodynamics':EventTermCfg(func=mdp.install_hydrodynamics,mode='startup'),
        },
        rewards={
            'oar_tracking':RewardTermCfg(func=mdp.track_oars,weight=1.0),
            'joint_velocity':RewardTermCfg(func=mdp.joint_velocity_cost,weight=-0.01),
            'joint_torque':RewardTermCfg(func=mdp.joint_torque_cost,weight=-0.001),
            'action_rate':RewardTermCfg(func=mdp.action_rate_cost,weight=-0.005),
            'collision':RewardTermCfg(func=mdp.collision_cost,weight=-0.10),
        },
        terminations={
            'capsized':TerminationTermCfg(func=mdp.capsized),
            'invalid_state':TerminationTermCfg(func=mdp.invalid_state),
            'time_out':TerminationTermCfg(func=common.time_out,time_out=True),
        },
        metrics={
            'tracking_rmse_deg':MetricsTermCfg(func=mdp.tracking_rmse_deg),
            'torque_saturation_fraction':MetricsTermCfg(func=mdp.torque_saturation),
        },
    )


def rl_cfg(num_envs=2048):
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    model = dict(class_name='rowing_mjlab.networks:TemporalModel',hidden_dims=(256,128,128),
                 activation='elu',obs_normalization=True)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(**model,distribution_cfg={'class_name':'GaussianDistribution',
                           'init_std':0.3,'std_type':'scalar'}),
        critic=RslRlModelCfg(**model),
        algorithm=RslRlPpoAlgorithmCfg(class_name='rowing_mjlab.storage:HistoryPPO',
            num_learning_epochs=4,num_mini_batches=max(1,num_envs*24//512),
            learning_rate=3e-4,schedule='adaptive',gamma=0.99,lam=0.95,
            entropy_coef=0.005,desired_kl=0.01,max_grad_norm=1.0,
            value_loss_coef=1.0,use_clipped_value_loss=True,clip_param=0.2),
        experiment_name='h2_rowing_imitation',num_steps_per_env=24,
        max_iterations=10000,save_interval=100,
        obs_groups={'actor':['actor'],'critic':['actor','privileged']},
        logger='tensorboard',clip_actions=1.0,
    )
