# Base class for one episode unit ("arena"). A Godot process holds many arenas spaced far
# apart; each has its own RNG, agents, episode clock and same-step auto-reset.
extends Node3D

const Kit = preload("res://common/kit.gd")
const Agent = preload("res://common/agent.gd")

const SPACING := 150.0     # distance between arena copies (>> camera far plane)

var arena_id := 0
var rng := RandomNumberGenerator.new()
var params := {}           # scenario/curriculum params (set from Python via set_params)
var agents: Array = []
var pov_size := Vector2i(64, 64)
var t := 0                 # ticks since episode start
var max_ticks := 1800
var ended := false
var ended_truncated := false
var skip_rest := false     # episode ended mid-decision: idle until next decision
var stats := {}            # per-episode stats reported in info
var pretty := false


func setup(p_id: int, p_seed: int, p_params: Dictionary, p_pov: Vector2i, p_pretty: bool) -> void:
	arena_id = p_id
	rng.seed = hash([p_seed, p_id])
	params = p_params.duplicate(true)
	pov_size = p_pov
	pretty = p_pretty
	build()
	do_reset()


func spawn_agent(team: int, policy: String, color: Color) -> Node:
	var a = Agent.new()
	a.arena = self
	a.team = team
	a.policy_name = policy
	add_child(a)
	a.setup(color, pov_size, agents.size())
	agents.append(a)
	return a


func do_reset() -> void:
	t = 0
	ended = false
	ended_truncated = false
	stats = {}
	for a in agents:
		a.ep_return = 0.0
	reset_episode()
	for a in agents:
		a.sync_camera()


func end_episode(truncated := false) -> void:
	if ended:
		return
	ended = true
	ended_truncated = truncated


func time_left_frac() -> float:
	return 1.0 - float(t) / float(max_ticks)


func episode_info(agent) -> Dictionary:
	var d := stats.duplicate()
	d["ep_return"] = agent.ep_return
	d["ep_len"] = t
	d["truncated"] = ended_truncated
	d["policy"] = agent.policy_name
	d["team"] = agent.team
	return d


func vec_dim() -> int:
	return Agent.BASE_VEC_DIM + extra_vec_size()


func get_vec(agent) -> Array:
	return agent.base_vec() + extra_vec(agent)


# ---- helpers ---------------------------------------------------------------------------
func ray(from: Vector3, to: Vector3, mask: int, exclude: Array = []) -> Dictionary:
	var q := PhysicsRayQueryParameters3D.new()
	q.from = from
	q.to = to
	q.collision_mask = mask
	var ex: Array[RID] = []
	for r in exclude:
		ex.append(r)
	q.exclude = ex
	return get_world_3d().direct_space_state.intersect_ray(q)


static func yaw_toward(from: Vector3, to: Vector3) -> float:
	var d := to - from
	return atan2(-d.x, -d.z)


static func local_bearing(agent, world_dir: Vector3) -> float:
	# angle of a world direction relative to the agent's facing (0 = ahead, + = right)
	var l: Vector3 = agent.global_transform.basis.inverse() * world_dir
	return atan2(l.x, -l.z)


# ---- scenario hooks (override) -------------------------------------------------------
func build() -> void:
	pass

func reset_episode() -> void:
	pass

func pre_tick(_dt: float) -> void:
	pass

func post_tick(_dt: float) -> void:
	pass

func on_primary(_agent) -> void:
	pass

func extra_vec_size() -> int:
	return 0

func extra_vec(_agent) -> Array:
	return []

func replay_flags(_agent) -> int:
	return 0  # scenario-specific bits for the cinema replay (e.g. tagged / holding)

func set_params(p: Dictionary) -> void:
	for k in p:
		params[k] = p[k]

# Camera framing for screenshots / spectating, in arena-local coordinates.
func spectator_pose() -> Array:
	return [Vector3(0, 14, 16), Vector3(0, 0, 0)]

func topdown_extent() -> float:
	return 26.0

func topdown_center() -> Vector3:
	return Vector3.ZERO
