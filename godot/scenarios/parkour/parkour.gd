# Parkour: a procedurally generated course of floating platforms along -z. Difficulty d in [0,1]
# (params {"difficulty": d} or {"difficulty_range": [lo, hi]}) scales the gap size, height
# changes, lateral offsets, the number of platforms, the chance of moving platforms, and a
# button that opens a door blocking the path.
# Rewards: +0.1/m of new forward progress, +0.5 per new platform, +0.5 button, +5 goal, -1 fall.
extends "res://common/arena_base.gd"

const MAX_PLAT := 14
const MAX_MOVING := 4
const HALF := 1.5            # regular platform half-size (3 x 0.6 x 3)

var agent
var plats: Array = []        # StaticBody3D pool
var movers: Array = []       # AnimatableBody3D pool
var start_plat: StaticBody3D
var goal_plat: StaticBody3D
var door: StaticBody3D
var button: MeshInstance3D
var course: Array = []       # [{body, top: Vector3 (center of top surface), moving, amp, phase, base}]
var n_jumps := 0
var best_z := 0.0
var last_idx := 0
var door_idx := -1
var door_open := true
var difficulty := 0.2
var min_top := 0.0


func build() -> void:
	max_ticks = 30 * 60
	var pm := Kit.checker_mat(Color(0.55, 0.6, 0.68), Color(0.48, 0.53, 0.6), 0.75)
	for i in MAX_PLAT:
		plats.append(Kit.static_box(self, Vector3(3, 0.6, 3), Vector3(0, -60, 0), pm))
	for i in MAX_MOVING:
		var b := AnimatableBody3D.new()
		b.sync_to_physics = true
		b.collision_layer = Kit.LAYER_WORLD
		b.collision_mask = 0
		var cs := CollisionShape3D.new()
		var sh := BoxShape3D.new()
		sh.size = Vector3(3, 0.6, 3)
		cs.shape = sh
		b.add_child(cs)
		b.add_child(Kit.mesh_instance(Kit.box_mesh(Vector3(3, 0.6, 3)), Kit.checker_mat(Color(0.95, 0.6, 0.2), Color(0.85, 0.5, 0.15), 0.75)))
		add_child(b)
		movers.append(b)
		_move_static(b, Vector3(0, -60, 0))
		_set_active(b, false)
	start_plat = Kit.static_box(self, Vector3(4, 0.6, 4), Vector3(0, -0.3, 0), Kit.checker_mat(Color(0.85, 0.85, 0.85), Color(0.75, 0.75, 0.75), 0.75))
	goal_plat = Kit.static_box(self, Vector3(4, 0.6, 4), Vector3(0, -60, 0), Kit.checker_mat(Kit.C_GOAL, Kit.C_GOAL.darkened(0.2), 0.75))
	door = Kit.static_box(self, Vector3(3.4, 3.0, 0.3), Vector3(0, -60, 0), Kit.mat(Color(0.8, 0.2, 0.25)))
	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.45
	cyl.bottom_radius = 0.5
	cyl.height = 0.12
	button = Kit.mesh_instance(cyl, Kit.mat(Kit.C_BUTTON, true))
	add_child(button)
	var ground := Kit.mesh_instance(Kit.box_mesh(Vector3(80, 1, 160)), Kit.checker_mat(Color(0.22, 0.3, 0.26), Color(0.2, 0.27, 0.23), 8.0))
	ground.position = Vector3(0, -25, -50)
	add_child(ground)
	agent = spawn_agent(0, "parkour", Kit.C_AGENT)


func _move_static(b: Node3D, local_pos: Vector3) -> void:
	# teleport. AnimatableBody3D with sync_to_physics ignores direct position writes, so turn it
	# off for the jump and back on afterwards (then per-tick motion carries riders smoothly).
	var anim := b as AnimatableBody3D
	if anim != null:
		anim.sync_to_physics = false
	b.position = local_pos
	PhysicsServer3D.body_set_state(b.get_rid(), PhysicsServer3D.BODY_STATE_TRANSFORM, b.global_transform)
	if anim != null:
		anim.sync_to_physics = true


func _set_active(b: CollisionObject3D, on: bool) -> void:
	b.visible = on
	b.collision_layer = Kit.LAYER_WORLD if on else 0


func reset_episode() -> void:
	if params.has("difficulty_range"):
		var r: Array = params["difficulty_range"]
		difficulty = rng.randf_range(float(r[0]), float(r[1]))
	else:
		difficulty = float(params.get("difficulty", 0.2))
	var d := difficulty
	max_ticks = int(params.get("seconds", 30)) * 60
	for p in plats:
		_move_static(p, Vector3(0, -60, 0))
	for m in movers:
		_move_static(m, Vector3(0, -60, 0))
		_set_active(m, false)
	_move_static(door, Vector3(0, -60, 0))
	button.visible = false
	course.clear()
	course.append({"body": start_plat, "top": Vector3(0, 0, 0), "moving": false, "half": 2.0})
	n_jumps = clampi(4 + int(round(8.0 * d)) + rng.randi_range(0, 1), 3, MAX_PLAT)
	var pos := Vector3.ZERO
	var prev_half := 2.0
	var n_moving := 0
	min_top = 0.0
	for i in range(1, n_jumps + 1):
		var gap := rng.randf_range(0.3, 0.6 + 2.2 * d)
		var dy := rng.randf_range(-0.3 - 0.6 * d, 0.2 + 0.65 * d)
		var dx := rng.randf_range(-1.6 * d, 1.6 * d)
		var moving := i > 1 and n_moving < MAX_MOVING and rng.randf() < 0.35 * d
		pos = Vector3(clampf(pos.x + dx, -6, 6), pos.y + dy, pos.z - (prev_half + gap + HALF))
		var body: Node3D
		if moving:
			body = movers[n_moving]
			_set_active(body, true)
			n_moving += 1
		else:
			body = plats[i - 1]
		_move_static(body, pos + Vector3(0, -0.3, 0))
		course.append({"body": body, "top": pos, "moving": moving, "half": HALF, "base_x": pos.x,
			"amp": rng.randf_range(1.0, 1.0 + 1.5 * d), "phase": rng.randf_range(0, TAU), "w": rng.randf_range(1.2, 2.2)})
		prev_half = HALF
		min_top = minf(min_top, pos.y)
	var gap_g := rng.randf_range(0.3, 0.6 + 1.5 * d)
	pos = Vector3(pos.x, pos.y + rng.randf_range(-0.2, 0.4 * d), pos.z - (HALF + gap_g + 2.0))
	_move_static(goal_plat, pos + Vector3(0, -0.3, 0))
	course.append({"body": goal_plat, "top": pos, "moving": false, "half": 2.0})
	min_top = minf(min_top, pos.y)
	# button + door
	door_idx = -1
	door_open = true
	if n_jumps >= 4 and rng.randf() < float(params.get("door_prob", 0.6 * d)):
		var k := rng.randi_range(1, n_jumps - 2)
		if not course[k]["moving"] and not course[k + 1]["moving"]:
			door_idx = k + 1
			door_open = false
			var bt: Vector3 = course[k]["top"]
			button.position = bt + Vector3(rng.randf_range(-0.9, 0.9), 0.06, rng.randf_range(-0.6, 0.6))
			button.visible = true
			var nt: Vector3 = course[k + 1]["top"]
			_move_static(door, nt + Vector3(0, 1.5, HALF - 0.15))
	agent.place(Vector3(0, 0.05, 0.5), rng.randf_range(-0.3, 0.3))
	best_z = 0.0
	last_idx = 0
	stats = {"platforms": 0, "n_platforms": n_jumps + 1, "win": 0, "fell": 0, "difficulty": difficulty, "door": 1 if door_idx > 0 else 0, "button": 0}


func pre_tick(_dt: float) -> void:
	if t == 30 and RLSync.args.has("debug_movers"):
		for m in movers:
			print("MOVER ", m.position, " visible=", m.visible, " in_course=", course.any(func(c): return c["body"] == m))
	var time := float(t) / 60.0
	for c in course:
		if c["moving"]:
			var b: AnimatableBody3D = c["body"]
			var x: float = c["base_x"] + c["amp"] * sin(c["phase"] + c["w"] * time)
			b.position = Vector3(x, c["top"].y - 0.3, c["top"].z)


func post_tick(_dt: float) -> void:
	var p: Vector3 = agent.position
	# forward progress
	if -p.z > best_z:
		agent.add_reward(0.1 * (-p.z - best_z))
		best_z = -p.z
	# platform landed on
	if agent.is_on_floor():
		for i in range(last_idx + 1, course.size()):
			var c: Dictionary = course[i]
			var b: Node3D = c["body"]
			var cx: float = b.position.x
			var top: Vector3 = c["top"]
			var h: float = c["half"]
			if absf(p.x - cx) < h + 0.3 and absf(p.z - top.z) < h + 0.3 and absf(p.y - top.y) < 0.4:
				last_idx = i
				stats["platforms"] = i
				agent.add_reward(0.5)
				if i == course.size() - 1:
					agent.add_reward(5.0)
					stats["win"] = 1
					end_episode()
					return
				break
		if not door_open and button.visible and Vector2(p.x - button.position.x, p.z - button.position.z).length() < 0.75:
			door_open = true
			button.visible = false
			stats["button"] = 1
			agent.add_reward(0.5)
			_move_static(door, Vector3(0, -60, 0))
	if p.y < min_top - 5.0:
		agent.add_reward(-1.0)
		stats["fell"] = 1
		end_episode()


func extra_vec_size() -> int:
	return 2


func extra_vec(_a) -> Array:
	return [float(last_idx) / float(max(1, course.size() - 1)), 1.0 if door_open else 0.0]


func _course_mid() -> Vector3:
	var end_top: Vector3 = course[course.size() - 1]["top"] if course.size() > 0 else Vector3(0, 0, -20)
	return Vector3(0, 0, end_top.z / 2.0)


func spectator_pose() -> Array:
	var m := _course_mid()
	return [m + Vector3(18, 12, 10), m + Vector3(0, -1, 2)]


func topdown_extent() -> float:
	return absf(_course_mid().z) * 2.0 + 8.0


func topdown_center() -> Vector3:
	return _course_mid()
