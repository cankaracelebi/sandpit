# Sumo (1v1): two agents on a round platform floating over a void. Primary = shove (impulse on an
# opponent in front, within reach, with cooldown); walking into the opponent also pushes. Leaving
# the ring (falling) loses. Symmetric self-play: both agents share policy "sumo".
# Rewards: +1 win / -1 loss / 0 draw (timeout), plus small potential-based edge shaping.
extends "res://common/arena_base.gd"

const R := 6.0
const SHOVE_RANGE := 2.0
const SHOVE_COS := 0.75     # ~41 degrees
const SHOVE_SPEED := 9.0
const FALL_Y := -2.0

var a0
var a1
var prev_d := [0.0, 0.0]


func build() -> void:
	Kit.static_cylinder(self, R, 1.0, Vector3(0, -0.5, 0), Kit.checker_mat(Color(0.82, 0.8, 0.74), Color(0.74, 0.72, 0.66), 1.0))
	# red rim marks the edge
	var rim := CylinderMesh.new()
	rim.top_radius = R + 0.02
	rim.bottom_radius = R + 0.02
	rim.height = 0.9
	rim.radial_segments = 48
	var rim_mi := Kit.mesh_instance(rim, Kit.mat(Color(0.75, 0.15, 0.15)))
	rim_mi.position = Vector3(0, -0.56, 0)
	add_child(rim_mi)
	# far ground for depth cues (no collision)
	var ground := Kit.mesh_instance(Kit.box_mesh(Vector3(120, 1, 120)), Kit.checker_mat(Color(0.25, 0.35, 0.28), Color(0.22, 0.3, 0.25), 6.0))
	ground.position = Vector3(0, -30, 0)
	add_child(ground)
	a0 = spawn_agent(0, "sumo", Kit.C_TEAM[0])
	a1 = spawn_agent(1, "sumo", Kit.C_TEAM[1])
	for a in [a0, a1]:
		a.cooldown_ticks = 36
		var hands := Kit.mesh_instance(Kit.box_mesh(Vector3(0.7, 0.15, 0.25)), Kit.mat(Color(0.95, 0.85, 0.75)))
		hands.position = Vector3(0, -0.45, -0.45)
		a.head.add_child(hands)
		a.add_self_mesh(hands)
	max_ticks = 20 * 60


func reset_episode() -> void:
	var ang := rng.randf_range(-PI, PI)
	var d := rng.randf_range(2.2, 3.5)
	var p0 := Vector3(cos(ang), 0, sin(ang)) * d
	var p1 := -p0.rotated(Vector3.UP, rng.randf_range(-0.4, 0.4))
	a0.place(Vector3(p0.x, 0.05, p0.z), yaw_toward(p0, p1) + rng.randf_range(-0.6, 0.6))
	a1.place(Vector3(p1.x, 0.05, p1.z), yaw_toward(p1, p0) + rng.randf_range(-0.6, 0.6))
	prev_d = [_center_dist(a0), _center_dist(a1)]
	stats = {"shoves": [0, 0], "shove_hits": [0, 0], "result": [0, 0]}


func _center_dist(a) -> float:
	var p: Vector3 = a.position
	return Vector2(p.x, p.z).length()


func _other(a):
	return a1 if a == a0 else a0


func on_primary(a) -> void:
	var o = _other(a)
	var i := 0 if a == a0 else 1
	stats["shoves"][i] += 1
	var to: Vector3 = o.global_position - a.global_position
	to.y = 0.0
	if to.length() > SHOVE_RANGE or to.length() < 0.01:
		return
	var f: Vector3 = a.forward()
	f.y = 0.0
	if f.normalized().dot(to.normalized()) < SHOVE_COS:
		return
	stats["shove_hits"][i] += 1
	o.knockback += to.normalized() * SHOVE_SPEED
	o.velocity.y = 3.0


func post_tick(_dt: float) -> void:
	# body pushing: walking into the opponent transfers momentum
	var d: Vector3 = a1.global_position - a0.global_position
	d.y = 0.0
	var dist := d.length()
	if dist < 0.95 and dist > 0.001:
		var n := d / dist
		var v0 := maxf(0.0, a0.walk_vel.dot(n))
		var v1 := maxf(0.0, a1.walk_vel.dot(-n))
		a1.knockback += n * v0 * 0.12
		a0.knockback -= n * v1 * 0.12
	# potential-based edge shaping: push the opponent out, stay central yourself
	var cd := [_center_dist(a0), _center_dist(a1)]
	for i in 2:
		var me = a0 if i == 0 else a1
		var j := 1 - i
		me.add_reward(0.05 * ((cd[j] - prev_d[j]) - (cd[i] - prev_d[i])) / R)
	prev_d = cd
	var f0: bool = a0.global_position.y - global_position.y < FALL_Y
	var f1: bool = a1.global_position.y - global_position.y < FALL_Y
	if f0 or f1:
		var r0 := 0.0
		if f0 and not f1:
			r0 = -1.0
		elif f1 and not f0:
			r0 = 1.0
		a0.add_reward(r0)
		a1.add_reward(-r0)
		stats["result"] = [int(r0), int(-r0)]
		end_episode()


func episode_info(agent) -> Dictionary:
	var i := 0 if agent == a0 else 1
	var d := {
		"ep_return": agent.ep_return, "ep_len": t, "truncated": ended_truncated, "policy": agent.policy_name,
		"team": agent.team, "result": stats["result"][i], "win": 1 if stats["result"][i] > 0 else 0,
		"shoves": stats["shoves"][i], "shove_hits": stats["shove_hits"][i],
	}
	return d


func extra_vec_size() -> int:
	return 3


func extra_vec(a) -> Array:
	var p: Vector3 = a.position
	var to_center := Vector3(-p.x, 0, -p.z)
	var b := local_bearing(a, to_center) if to_center.length() > 0.01 else 0.0
	return [_center_dist(a) / R, sin(b), cos(b)]


func spectator_pose() -> Array:
	return [Vector3(0, 9, 12), Vector3(0, -0.5, 0)]


func topdown_extent() -> float:
	return 2 * R + 4
