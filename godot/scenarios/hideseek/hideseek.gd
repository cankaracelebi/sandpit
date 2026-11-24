# Tag and Hide & Seek (two modes, one arena). Hiders = team 0 (policy "hider"), seekers = team 1
# (policy "seeker"). The arena has a corner room with a doorway, movable boxes and a ramp
# (RigidBody3D). Agents push props by walking into them and can grab/carry them with primary.
#
# mode "tag" (default 1v1): seekers are frozen during a short prep phase. Seeker primary = tag (a
#   hider in reach and in front). Tagged hiders freeze. Episode ends when all hiders are tagged
#   or on timeout. Rewards: tag +1 seeker team / -1 tagged hider; +-0.001 per tick survival.
# mode "hideseek" (default 2v2): seekers are frozen and blind during a 4 s prep phase. Afterwards,
#   every tick: if any hider is visible to any seeker (in FOV + line of sight), hiders -0.005 and
#   seekers +0.005, otherwise the reverse. Primary = grab/release for everyone.
extends "res://common/arena_base.gd"

const ROOM := 18.0
const FOV_COS := 0.70        # ~45.6 deg half-angle (camera is 90 deg)
const GRAB_RANGE := 2.0
var tag_range := 2.0

var mode := "tag"
var hiders: Array = []
var seekers: Array = []
var props: Array = []
var layout: Node3D
var prep_ticks := 90
var holding := {}            # agent -> RigidBody3D
var tagged := {}             # agent -> bool
var hidden_ticks := 0
var seek_ticks := 0
var prev_dist := {}           # tag shaping: agent -> distance to nearest opponent


func build() -> void:
	mode = str(params.get("mode", "tag"))
	var n_h := int(params.get("n_hiders", 1 if mode == "tag" else 2))
	var n_s := int(params.get("n_seekers", 1 if mode == "tag" else 2))
	Kit.walled_room(self, ROOM, ROOM, 3.0, Kit.checker_mat(Color(0.42, 0.44, 0.40), Color(0.36, 0.38, 0.34)), Kit.mat(Kit.C_WALL))
	# corner room with a doorway; the whole layout node is rotated per episode
	layout = Node3D.new()
	add_child(layout)
	var wm := Kit.mat(Color(0.7, 0.66, 0.6))
	Kit.static_box(layout, Vector3(4.0, 3.0, 0.4), Vector3(4.0, 1.5, -2.0), wm)   # segment, door gap 2..? left of it
	Kit.static_box(layout, Vector3(0.4, 3.0, 4.5), Vector3(2.0, 1.5, -6.75), wm)
	Kit.static_box(layout, Vector3(1.0, 3.0, 0.4), Vector3(8.5, 1.5, -2.0), wm)
	Kit.static_box(layout, Vector3(2.5, 3.0, 0.4), Vector3(-4.5, 1.5, 3.5), wm)   # free-standing wall
	for i in 3:
		props.append(Kit.rigid_box(self, Vector3(1.2, 1.2, 1.2), Vector3(0, -30 - i * 3, 0), Kit.mat(Kit.C_BOX), 20.0))
	props.append(Kit.rigid_box(self, Vector3(2.4, 1.2, 1.2), Vector3(0, -45, 0), Kit.mat(Kit.C_BOX.darkened(0.15)), 35.0))
	props.append(Kit.rigid_ramp(self, Vector3(2.4, 1.2, 1.5), Vector3(0, -50, 0), Kit.mat(Kit.C_RAMP), 30.0))
	for i in n_h:
		hiders.append(spawn_agent(0, "hider", Kit.C_TEAM[0].lerp(Color(0.2, 0.85, 0.9), 0.35 * i)))
	for i in n_s:
		seekers.append(spawn_agent(1, "seeker", Kit.C_TEAM[1].lerp(Color(0.95, 0.25, 0.2), 0.35 * i)))
	for a in agents:
		a.cooldown_ticks = 20
	# tag is only winnable if the chaser is faster than an optimally fleeing runner
	tag_range = float(params.get("tag_range", 2.0))
	if mode == "tag":
		for a in seekers:
			a.move_speed *= float(params.get("seeker_speed", 1.25))


func _rand_point(margin := 1.5) -> Vector3:
	var h := ROOM / 2.0 - margin
	return Vector3(rng.randf_range(-h, h), 0.0, rng.randf_range(-h, h))


func reset_episode() -> void:
	mode = str(params.get("mode", mode))
	prep_ticks = int(params.get("prep_ticks", 90 if mode == "tag" else 240))
	max_ticks = int(params.get("seconds", 20)) * 60
	layout.rotation.y = rng.randi_range(0, 3) * PI / 2.0
	for b in layout.get_children():  # push the new wall transforms to physics right away
		PhysicsServer3D.body_set_state(b.get_rid(), PhysicsServer3D.BODY_STATE_TRANSFORM, b.global_transform)
	var taken: Array = []
	for pr in props:
		var p := _free_point(taken, 2.2, 2.0)
		taken.append(p)
		var xf := Transform3D(Basis(Vector3.UP, rng.randf_range(-PI, PI)), to_global(Vector3(p.x, 0.62, p.z)))
		Kit.teleport_rigid(pr, xf)
	for a in hiders:
		var p := _free_point(taken, 1.6, 1.5)
		taken.append(p)
		a.place(Vector3(p.x, 0.05, p.z), rng.randf_range(-PI, PI))
	for a in seekers:
		var p := _free_point(taken, 4.0, 1.5)
		taken.append(p)
		a.place(Vector3(p.x, 0.05, p.z), rng.randf_range(-PI, PI))
		a.frozen = true
		a.blind = mode == "hideseek"
	holding.clear()
	tagged.clear()
	prev_dist.clear()
	hidden_ticks = 0
	seek_ticks = 0
	stats = {"tags": 0, "hidden_frac": 0.0, "grabs": 0}


func _free_point(taken: Array, min_d: float, margin: float) -> Vector3:
	var p := _rand_point(margin)
	for attempt in 60:
		var ok := true
		for q in taken:
			if p.distance_to(q) < min_d:
				ok = false
				break
		if ok and _not_in_walls(p):
			return p
		p = _rand_point(margin)
	return p


func _not_in_walls(p: Vector3) -> bool:
	var g := to_global(Vector3(p.x, 1.0, p.z))
	var q := PhysicsPointQueryParameters3D.new()
	q.position = g
	q.collision_mask = Kit.LAYER_WORLD
	return get_world_3d().direct_space_state.intersect_point(q, 1).is_empty()


func _prep_left() -> float:
	return maxf(0.0, float(prep_ticks - t) / float(max(1, prep_ticks)))


func pre_tick(_dt: float) -> void:
	if t == prep_ticks:
		for a in seekers:
			a.frozen = false
			a.blind = false
	# carried props follow a hold point in front of the carrier
	for a in holding.keys():
		var rb: RigidBody3D = holding[a]
		var target: Vector3 = a.eye_position() + a.forward() * 1.7 + Vector3(0, -0.7, 0)
		var delta := target - rb.global_position
		if delta.length() > 3.0 or a.frozen:
			holding.erase(a)
			continue
		rb.linear_velocity = (delta * 10.0).limit_length(8.0)
		rb.angular_velocity *= 0.7


func on_primary(a) -> void:
	if mode == "tag" and a.team == 1:
		_try_tag(a)
		return
	if holding.has(a):
		holding.erase(a)
		return
	var best: RigidBody3D = null
	var best_d := GRAB_RANGE
	for pr in props:
		var to: Vector3 = pr.global_position - a.global_position
		to.y = 0.0
		var d := to.length()
		if d < best_d and d > 0.01 and a.forward().dot(to / d) > 0.5 and not holding.values().has(pr):
			best = pr
			best_d = d
	if best != null:
		holding[a] = best
		stats["grabs"] += 1


func _try_tag(a) -> void:
	for h in hiders:
		if tagged.get(h, false):
			continue
		var to: Vector3 = h.global_position - a.global_position
		to.y = 0.0
		var d := to.length()
		if d < tag_range and (d < 0.6 or a.forward().dot(to / d) > 0.5):
			tagged[h] = true
			h.frozen = true
			h.set_color(Color(0.45, 0.45, 0.5))
			holding.erase(h)
			stats["tags"] += 1
			h.add_reward(-1.0)
			for s in seekers:
				s.add_reward(1.0)
	var all_tagged := true
	for h in hiders:
		all_tagged = all_tagged and tagged.get(h, false)
	if all_tagged:
		end_episode()


func _seen(s, h) -> bool:
	var eye: Vector3 = s.eye_position()
	var f: Vector3 = s.aim_direction()
	for off in [1.45, 0.9]:
		var tp: Vector3 = h.global_position + Vector3(0, off, 0)
		var to := tp - eye
		var d := to.length()
		if d > 30.0 or f.dot(to / d) < FOV_COS:
			continue
		var hit := ray(eye, tp, Kit.LAYER_WORLD | Kit.LAYER_PROP)
		if hit.is_empty():
			return true
	return false


func post_tick(_dt: float) -> void:
	if t < prep_ticks:
		return
	seek_ticks += 1
	if mode == "tag":
		for h in hiders:
			if not tagged.get(h, false):
				h.add_reward(0.001)
		for s in seekers:
			s.add_reward(-0.001)
		# potential-based distance shaping (seekers close in, hiders get away)
		var k := float(params.get("dist_shaping", 0.05))
		for a in agents:
			var opp: Array = hiders if a.team == 1 else seekers
			var best := 1e9
			for o in opp:
				if not tagged.get(o, false):
					best = minf(best, a.global_position.distance_to(o.global_position))
			if best < 1e8:
				if prev_dist.has(a):
					var dd: float = prev_dist[a] - best
					a.add_reward(k * dd if a.team == 1 else -k * dd)
				prev_dist[a] = best
	else:
		var any_seen := false
		for s in seekers:
			for h in hiders:
				if _seen(s, h):
					any_seen = true
					break
			if any_seen:
				break
		if not any_seen:
			hidden_ticks += 1
		var r := -0.005 if any_seen else 0.005
		for h in hiders:
			h.add_reward(r)
		for s in seekers:
			s.add_reward(-r)
	stats["hidden_frac"] = float(hidden_ticks) / float(max(1, seek_ticks))


func do_reset() -> void:
	for h in hiders:
		h.set_color(Kit.C_TEAM[0].lerp(Color(0.2, 0.85, 0.9), 0.35 * hiders.find(h)))
	super.do_reset()


func episode_info(agent) -> Dictionary:
	var d := super.episode_info(agent)
	d["role"] = "hider" if agent.team == 0 else "seeker"
	var win := 0
	if mode == "tag":
		var all_tagged := true
		for h in hiders:
			all_tagged = all_tagged and tagged.get(h, false)
		win = 1 if (all_tagged == (agent.team == 1)) else 0
	else:
		var hid: bool = float(stats["hidden_frac"]) > 0.5
		win = 1 if (hid == (agent.team == 0)) else 0
	d["win"] = win
	return d


func replay_flags(a) -> int:
	return (1 if tagged.get(a, false) else 0) | (2 if holding.has(a) else 0)


func extra_vec_size() -> int:
	return 4


func extra_vec(a) -> Array:
	return [float(a.team), _prep_left(), 1.0 if holding.has(a) else 0.0, 1.0 if tagged.get(a, false) else 0.0]


func spectator_pose() -> Array:
	return [Vector3(0, 15, 15), Vector3(0, 0, 0)]


func topdown_extent() -> float:
	return ROOM + 2.0
