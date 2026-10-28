"""mjlab action, observation, reward, reset and startup terms."""
from dataclasses import dataclass
import json
import torch
import warp as wp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from hydrodynamics.models import ROOT
from .asset import OAR_NAMES, h2_names
from .hydro import WarpHydrodynamics
from .reference import RowingReference, IMITATION_NAMES
from .blade_area import BladeAreaReference
from .trunk import TrunkReference, trunk_vertical_angle, angular_error
from .symmetry import BilateralSymmetry


class RowingState:
    def __init__(self,env,hull_radiation=True,blade_inertia=True):
        self.env = env
        model = env.sim.mj_model
        device = env.device
        self.h2_names = h2_names()
        self.policy_names = tuple(n for n in self.h2_names if not n.startswith('head_'))
        def joints(names,field):
            return torch.tensor([getattr(model,field)[model.joint('rowing/'+n).id]
                                 for n in names],device=device,dtype=torch.long)
        self.h2_q = joints(self.h2_names,'jnt_qposadr')
        self.h2_v = joints(self.h2_names,'jnt_dofadr')
        self.policy_q = joints(self.policy_names,'jnt_qposadr')
        self.policy_v = joints(self.policy_names,'jnt_dofadr')
        self.oar_q = joints(OAR_NAMES,'jnt_qposadr')
        self.boat = int(model.body('rowing/boat').id)
        self.imu = int(model.body('rowing/imu_in_pelvis').id)
        self.torso = int(model.body('rowing/torso_link').id)
        self.q0 = torch.tensor(model.qpos0.copy(),device=device,dtype=torch.float32)
        self.symmetry = BilateralSymmetry(model,self.h2_names,device,prefix='rowing/')
        self.controlled_local = torch.tensor([self.h2_names.index(n) for n in self.policy_names],device=device)
        joint_ids = [model.joint('rowing/'+n).id for n in self.h2_names]
        policy_ids = [model.joint('rowing/'+n).id for n in self.policy_names]
        self.limits = torch.tensor(model.jnt_range[joint_ids].copy(),device=device,dtype=torch.float32)
        self.torque_limits = torch.tensor(model.jnt_actfrcrange[policy_ids,1].copy(),device=device,dtype=torch.float32)
        self.imitation_q = joints(IMITATION_NAMES,'jnt_qposadr')
        self.ref = RowingReference(device,model,prefix='rowing/')
        self.initial_imitation = self.q0[self.imitation_q]
        self.initial_oars = self.q0[self.oar_q]
        self.phase0 = self.ref.nearest(self.initial_oars)
        self.trunk_ref = TrunkReference(device)
        import mujoco
        initial_data = mujoco.MjData(model)
        initial_data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, initial_data)
        self.initial_trunk = trunk_vertical_angle(
            torch.tensor(initial_data.xmat[self.torso].reshape(3,3).copy(), device=device, dtype=torch.float32),
            torch.tensor(initial_data.xmat[self.boat].reshape(3,3).copy(), device=device, dtype=torch.float32))
        self.hydro = WarpHydrodynamics(model,env.num_envs,device,prefix='rowing/',
            hull_radiation=hull_radiation,blade_inertia=blade_inertia)
        provenance = json.loads((ROOT/'data/rowing_reference_provenance.json').read_text())
        wp.to_torch(self.hydro.water)[:,0] = provenance['water_level_m_at_repository_reset']
        self.water_level = wp.to_torch(self.hydro.water)[:,0]
        self.blade_area = BladeAreaReference(model,self.ref,
            provenance['water_level_m_at_repository_reset'],device,prefix='rowing/')
        self.loads = wp.to_torch(self.hydro.loads)
        # Resolve entity-local indices independently from model joint order.
        entity = env.scene['rowing']
        self.entity_joint_ids,_ = entity.find_joints(self.h2_names,preserve_order=True)

    def reference(self):
        return self.ref.target(self.env.episode_length_buf*self.env.step_dt,self.phase0,self.initial_imitation)


@dataclass(kw_only=True)
class RowingActionCfg(ActionTermCfg):
    scale: float = 1.0  # Fraction of available joint travel on each side of q0.
    hull_radiation: bool = True
    blade_inertia: bool = True

    def build(self,env):
        return RowingAction(self,env)


class RowingAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        env.rowing = RowingState(env,hull_radiation=cfg.hull_radiation,blade_inertia=cfg.blade_inertia)
        self.state = env.rowing
        self._raw = torch.zeros(env.num_envs,self.action_dim,device=env.device)
        self.targets = self.state.q0[self.state.h2_q].expand(env.num_envs,-1).clone()
        # Alias stays current because targets are updated/reset in place.
        self.state.commanded_targets = self.targets

    @property
    def action_dim(self):
        return len(self.state.policy_names) if hasattr(self,'state') else 29

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self,actions):
        self._raw.copy_(actions)
        s = self.state
        a = actions.clamp(-1,1)*self.cfg.scale
        q0 = s.q0[s.policy_q]
        limits = s.limits[s.controlled_local]
        travel = torch.where(a >= 0,limits[:,1]-q0,q0-limits[:,0])
        self.targets[:,s.controlled_local] = q0+a*travel
        self.targets.clamp_(s.limits[:,0],s.limits[:,1])
        # Head columns retain the repository target; no policy action reaches them.

    def apply_actions(self):
        self._entity.set_joint_position_target(self.targets,joint_ids=self.state.entity_joint_ids)

    def reset(self,env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.targets[ids] = self.state.q0[self.state.h2_q]
        self.state.hydro.reset(env_ids)


def install_hydrodynamics(env,env_ids=None):
    if env.sim.wp_model.callback.control is not None:
        raise RuntimeError('A control callback already exists; do not silently replace it')
    env.sim.wp_model.callback.control = env.rowing.hydro
    wp.load_module(module='rowing_mjlab.hydro',device=env.device)
    if env.rowing.hydro.radiation is not None:
        wp.load_module(module='rowing_mjlab.radiation',device=env.device)
    if env.rowing.hydro.blade_inertia is not None:
        wp.load_module(module='rowing_mjlab.blade_inertia',device=env.device)
    # Re-capture so step AND forward graphs include current-state hydrodynamics.
    env.sim.create_graph()


def onboard_proprio(env):
    s,d = env.rowing,env.sim.data
    rotation = d.xmat[:,s.imu].reshape(-1,3,3)
    gyro = torch.bmm(rotation.transpose(1,2),d.cvel[:,s.imu,:3,None]).squeeze(-1)
    gravity = -rotation[:,2,:]
    return torch.cat((d.qpos[:,s.h2_q]-s.q0[s.h2_q],d.qvel[:,s.h2_v],gyro,gravity),dim=-1)


def onboard_command(env):
    phase,_,_,_ = env.rowing.reference()
    return torch.stack((torch.sin(2*torch.pi*phase),torch.cos(2*torch.pi*phase)),dim=-1)


def trunk_angles(env):
    s,d = env.rowing,env.sim.data
    actual = trunk_vertical_angle(d.xmat[:,s.torso].reshape(-1,3,3),
                                  d.xmat[:,s.boat].reshape(-1,3,3))
    phase,_,blend,_ = s.reference()
    target = s.trunk_ref.target(phase,blend,s.initial_trunk)
    return actual,target


def privileged_trunk(env):
    actual,target = trunk_angles(env)
    return torch.stack((actual,target),dim=-1)


def track_trunk(env):
    actual,target = trunk_angles(env)
    return torch.exp(-(angular_error(actual,target)/env.rowing.trunk_ref.sigma).square())


def trunk_error_deg(env):
    actual,target = trunk_angles(env)
    return torch.rad2deg(angular_error(actual,target).abs())


def last_action(env):
    return env.action_manager.action


def boat_world_velocity(env):
    s,d = env.rowing,env.sim.data
    cv = d.cvel[:,s.boat]
    # cvel translation is at root subtree COM; move to boat body COM.
    return cv[:,3:]+torch.cross(cv[:,:3],d.xipos[:,s.boat]-d.subtree_com[:,s.boat],dim=-1)


def boat_forward_speed(env):
    s,d = env.rowing,env.sim.data
    # Boat +X points behind the seated robot. Reward horizontal progress,
    # excluding heave even when the hull pitches; reverse remains negative.
    heading = d.xmat[:,s.boat].reshape(-1,3,3)[:,:2,0]
    heading = heading/torch.linalg.vector_norm(heading,dim=-1,keepdim=True).clamp_min(1e-8)
    return (boat_world_velocity(env)[:,:2]*heading).sum(dim=-1)


def boat_lateral_velocity_cost(env):
    """Squared horizontal speed perpendicular to the hull heading, in m²/s²."""
    s,d = env.rowing,env.sim.data
    heading = d.xmat[:,s.boat].reshape(-1,3,3)[:,:2,0]
    heading = heading/torch.linalg.vector_norm(heading,dim=-1,keepdim=True).clamp_min(1e-8)
    velocity = boat_world_velocity(env)[:,:2]
    return (heading[:,0]*velocity[:,1]-heading[:,1]*velocity[:,0]).square()


def boat_angular_velocity_cost(env):
    """Squared hull angular speed across roll, pitch and yaw, in rad²/s²."""
    return env.sim.data.cvel[:,env.rowing.boat,:3].square().sum(-1)


def privileged(env):
    s,d = env.rowing,env.sim.data
    rotation = d.xmat[:,s.boat].reshape(-1,3,3)
    v = boat_world_velocity(env)
    local_v = torch.bmm(rotation.transpose(1,2),v[...,None]).squeeze(-1)
    return torch.cat((local_v,s.loads.flatten(1),d.qpos[:,s.oar_q]),dim=-1)


def oar_tracking_scores(env):
    s = env.rowing
    target = s.reference()[-1][...,:6]
    error = ((env.sim.data.qpos[:,s.oar_q]-target)/s.ref.sigma).reshape(-1,2,3)
    return torch.exp(-error.square().mean(1))


def track_oar_sweep(env):
    return oar_tracking_scores(env)[:,0]


def track_oar_elevation(env):
    return oar_tracking_scores(env)[:,1]


def track_oar_feather(env):
    return oar_tracking_scores(env)[:,2]


def joint_torque_cost(env):
    s = env.rowing
    return (env.sim.data.qfrc_actuator[:,s.policy_v]/s.torque_limits).square().mean(-1)


def action_rate_cost(env):
    return (env.action_manager.action-env.action_manager.prev_action).square().mean(-1)


def bilateral_asymmetry_cost(env):
    s,d = env.rowing,env.sim.data
    return s.symmetry.cost(d.qpos,d.qvel,s.commanded_targets)


def bilateral_position_rmse_deg(env):
    s,d = env.rowing,env.sim.data
    position,_,_ = s.symmetry.errors(d.qpos,d.qvel,s.commanded_targets)
    return torch.rad2deg(position.square().mean(dim=-1).sqrt())


def bilateral_velocity_rmse_deg_s(env):
    s,d = env.rowing,env.sim.data
    _,velocity,_ = s.symmetry.errors(d.qpos,d.qvel,s.commanded_targets)
    return torch.rad2deg(velocity.square().mean(dim=-1).sqrt())


def bilateral_command_rmse_deg(env):
    s,d = env.rowing,env.sim.data
    _,_,command = s.symmetry.errors(d.qpos,d.qvel,s.commanded_targets)
    return torch.rad2deg(command.square().mean(dim=-1).sqrt())


def collision_cost(env):
    found = env.scene['forbidden_contact'].data.found
    return found.reshape(env.num_envs,-1).any(-1).float()


def boat_roll(env):
    # Use current root qpos, not pre-integration xmat.
    q = env.sim.data.qpos[:,3:7]  # wxyz, boat is the sole free root
    w,x,y,z = q.unbind(-1)
    return torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))


def roll_upright(env, sigma_deg=10.0):
    return torch.exp(-(boat_roll(env)/(sigma_deg*torch.pi/180)).square())


def boat_roll_abs_deg(env):
    return torch.rad2deg(boat_roll(env).abs())


def capsized(env):
    return boat_roll(env).abs() > torch.pi/3


def invalid_state(env):
    return ~(torch.isfinite(env.sim.data.qpos).all(-1)&torch.isfinite(env.sim.data.qvel).all(-1))


def tracking_rmse_deg(env):
    s = env.rowing
    return torch.rad2deg((env.sim.data.qpos[:,s.oar_q]-s.reference()[-1][...,:6]).square().mean(-1).sqrt())


def torque_saturation(env):
    s = env.rowing
    return (env.sim.data.qfrc_actuator[:,s.policy_v].abs() >= .98*s.torque_limits).float().mean(-1)


def track_seat(env):
    s = env.rowing
    error = (env.sim.data.qpos[:,s.imitation_q[6]]-s.reference()[-1][:,6])/s.ref.seat_sigma
    return torch.exp(-error.square())


def track_elbows(env):
    s = env.rowing
    error = (env.sim.data.qpos[:,s.imitation_q[7:]]-s.reference()[-1][:,7:])/s.ref.elbow_sigma
    return torch.exp(-error.square().mean(-1))


def seat_error_m(env):
    s = env.rowing
    return (env.sim.data.qpos[:,s.imitation_q[6]]-s.reference()[-1][:,6]).abs()


def elbow_rmse_deg(env):
    s = env.rowing
    return torch.rad2deg((env.sim.data.qpos[:,s.imitation_q[7:]]-s.reference()[-1][:,7:]).square().mean(-1).sqrt())


def blade_area_error(env):
    s = env.rowing
    phase,_,blend,_ = s.reference()
    return s.blade_area.actual(env.sim.data,s.water_level)-s.blade_area.target(phase,blend)


def track_blade_area(env):
    s = env.rowing
    error = blade_area_error(env)/(s.blade_area.full_area*s.blade_area.sigma)
    return torch.exp(-error.square().mean(-1))


def blade_area_rmse_m2(env):
    return blade_area_error(env).square().mean(-1).sqrt()
