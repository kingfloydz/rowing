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
from .networks import SEQUENCE_LENGTH


def env_cfg(num_envs=2048,play=False):
    return ManagerBasedRlEnvCfg(
        decimation=10, episode_length_s=40.0,
        scene=SceneCfg(num_envs=num_envs,env_spacing=0.0,entities={'rowing':entity_cfg()},
            sensors=(ContactSensorCfg(name='forbidden_contact',
                primary=ContactMatch(mode='geom',pattern='(h2|oar)_collision_.*',entity='rowing'),
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
            },history_length=SEQUENCE_LENGTH,flatten_history_dim=False,enable_corruption=not play),
            'privileged':ObservationGroupCfg(terms={
                'state':ObservationTermCfg(func=mdp.privileged),
                'trunk':ObservationTermCfg(func=mdp.privileged_trunk),
            },enable_corruption=False),
        },
        events={
            'reset_scene':EventTermCfg(func=common.reset_scene_to_default,mode='reset'),
            'hydrodynamics':EventTermCfg(func=mdp.install_hydrodynamics,mode='startup'),
        },
        rewards={
            'forward_speed':RewardTermCfg(func=mdp.boat_forward_speed,weight=0.6),
            'lateral_velocity':RewardTermCfg(func=mdp.boat_lateral_velocity_cost,weight=-0.6),
            'angular_velocity':RewardTermCfg(func=mdp.boat_angular_velocity_cost,weight=-0.6),
            'oar_sweep_tracking':RewardTermCfg(func=mdp.track_oar_sweep,weight=1.0),
            'oar_elevation_tracking':RewardTermCfg(func=mdp.track_oar_elevation,weight=1.0),
            'oar_feather_tracking':RewardTermCfg(func=mdp.track_oar_feather,weight=1.0),
            'seat_tracking':RewardTermCfg(func=mdp.track_seat,weight=0.5),
            'elbow_tracking':RewardTermCfg(func=mdp.track_elbows,weight=0.25),
            'trunk_tracking':RewardTermCfg(func=mdp.track_trunk,weight=0.6),
            'bilateral_asymmetry':RewardTermCfg(func=mdp.bilateral_asymmetry_cost,weight=-0.05),
            'roll_upright':RewardTermCfg(func=mdp.roll_upright,weight=0.6),
            'joint_torque':RewardTermCfg(func=mdp.joint_torque_cost,weight=-0.005),
            'action_rate':RewardTermCfg(func=mdp.action_rate_cost,weight=-0.025),
            'collision':RewardTermCfg(func=mdp.collision_cost,weight=-0.10),
        },
        terminations={
            'capsized':TerminationTermCfg(func=mdp.capsized),
            'invalid_state':TerminationTermCfg(func=mdp.invalid_state),
            'time_out':TerminationTermCfg(func=common.time_out,time_out=True),
        },
        metrics={
            'forward_speed_m_s':MetricsTermCfg(func=mdp.boat_forward_speed),
            'tracking_rmse_deg':MetricsTermCfg(func=mdp.tracking_rmse_deg),
            'blade_area_rmse_m2':MetricsTermCfg(func=mdp.blade_area_rmse_m2),
            'seat_error_m':MetricsTermCfg(func=mdp.seat_error_m),
            'elbow_rmse_deg':MetricsTermCfg(func=mdp.elbow_rmse_deg),
            'trunk_error_deg':MetricsTermCfg(func=mdp.trunk_error_deg),
            'boat_roll_abs_deg':MetricsTermCfg(func=mdp.boat_roll_abs_deg),
            'bilateral_position_rmse_deg':MetricsTermCfg(func=mdp.bilateral_position_rmse_deg),
            'bilateral_velocity_rmse_deg_s':MetricsTermCfg(func=mdp.bilateral_velocity_rmse_deg_s),
            'bilateral_command_rmse_deg':MetricsTermCfg(func=mdp.bilateral_command_rmse_deg),
            'torque_saturation_fraction':MetricsTermCfg(func=mdp.torque_saturation),
        },
    )


def rl_cfg(num_envs=2048):
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    rollout_steps = 32  # 0.64 s at 50 Hz; 2000/32 = 62.5 rollout blocks per episode.
    model = dict(class_name='rowing_mjlab.networks:TemporalModel',hidden_dims=(512,256,128),
                 activation='elu',obs_normalization=True)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(**model,distribution_cfg={'class_name':'GaussianDistribution',
                           'init_std':0.3,'std_type':'scalar'}),
        critic=RslRlModelCfg(**model),
        algorithm=RslRlPpoAlgorithmCfg(class_name='rowing_mjlab.storage:HistoryPPO',
            num_learning_epochs=4,num_mini_batches=max(1,num_envs*rollout_steps//512),
            learning_rate=3e-4,schedule='adaptive',gamma=0.99,lam=0.95,
            entropy_coef=0.005,desired_kl=0.01,max_grad_norm=1.0,
            value_loss_coef=1.0,use_clipped_value_loss=True,clip_param=0.2),
        experiment_name='h2_rowing_imitation',num_steps_per_env=rollout_steps,
        max_iterations=10000,save_interval=100,
        obs_groups={'actor':['actor'],'critic':['actor','privileged']},
        logger='tensorboard',clip_actions=1.0,
    )
