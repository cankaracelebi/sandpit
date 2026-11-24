# Shooter & Dodge: the agent has a hitscan gun. Scripted bots move around and fire slow, visible
# projectiles that can be dodged. A 5-stage curriculum is selected through params:
#   {"stage": k} or {"stage_probs": [p0..p4]}
# Rewards: +1 kill, +0.2 hit, -0.4*damage, -1 death, +2 clear, -0.005 per shot.
extends "res://common/arena_base.gd"

const ROOM := 24.0
const MAX_ENEMIES := 6
const MAX_PROJ := 24
const PROJ_R := 0.25
const STAGES := [
	{"name": "gallery", "n": [1, 3], "modes": ["static"], "fire": false, "spawn": "front", "hp": 1, "speed": 0.0, "time": 20},
	{"name": "movers", "n": [2, 3], "modes": ["strafe", "wander"], "fire": false, "spawn": "front", "hp": 1, "speed": 2.5, "time": 25},
	{"name": "dodge", "n": [1, 2], "modes": ["strafe"], "fire": true, "proj_speed": 10.0, "interval": 90, "noise": 0.10, "lead": 0.0, "spawn": "front", "hp": 1, "speed": 2.5, "time": 30},
	{"name": "arena", "n": [3, 4], "modes": ["static", "strafe", "wander", "chase"], "fire": true, "proj_speed": 14.0, "interval": 70, "noise": 0.06, "lead": 0.5, "spawn": "any", "hp": 2, "speed": 3.5, "time": 30},
	{"name": "hard", "n": [4, 6], "modes": ["strafe", "wander", "chase"], "fire": true, "proj_speed": 18.0, "interval": 50, "noise": 0.03, "lead": 1.0, "spawn": "any", "hp": 2, "speed": 4.5, "time": 30},
]
const OBSTACLE_SIZES := [Vector3(2, 2, 2), Vector3(3, 2, 1), Vector3(1, 2, 3), Vector3(2, 1.2, 2), Vector3(1.5, 2.5, 1.5), Vector3(4, 2, 1)]

var agent
var enemies: Array = []
var projectiles: Array = []
var obstacles: Array = []
var cfg: Dictionary = {}
var stage := 0
var dmg_dir := Vector2.ZERO
var dmg_timer := 0.0
var hit_timer := 0.0
var tracer: MeshInstance3D
var tracer_ticks := 0


func build() -> void:
	Kit.walled_room(self, ROOM, ROOM, 3.0, Kit.checker_mat(Kit.C_FLOOR_A, Kit.C_FLOOR_B), Kit.mat(Kit.C_WALL))
	for s in OBSTACLE_SIZES:
		obstacles.append(Kit.static_box(self, s, Vector3(0, -20, 0), Kit.mat(Kit.C_OBSTACLE)))
	agent = spawn_agent(0, "shooter", Kit.C_AGENT)
	var gun := Kit.mesh_instance(Kit.box_mesh(Vector3(0.1, 0.1, 0.5)), Kit.mat(Color(0.1, 0.1, 0.1)))
	gun.position = Vector3(0.28, -0.25, -0.3)
	agent.head.add_child(gun)
	agent.add_self_mesh(gun)
	for i in MAX_ENEMIES:
		enemies.append(_make_enemy(i))
	for i in MAX_PROJ:
		var m := Kit.mesh_instance(Kit.sphere_mesh(PROJ_R), Kit.mat(Kit.C_PROJ, true))
		add_child(m)
		m.visible = false
		projectiles.append({"mesh": m, "pos": Vector3.ZERO, "vel": Vector3.ZERO, "alive": false, "ttl": 0})
	tracer = Kit.mesh_instance(Kit.box_mesh(Vector3(0.04, 0.04, 1.0)), Kit.mat(Color(1.0, 0.95, 0.6), true))
	tracer.visible = false
	add_child(tracer)


func _make_enemy(i: int) -> Dictionary:
	var b := CharacterBody3D.new()
	b.motion_mode = CharacterBody3D.MOTION_MODE_FLOATING
	b.collision_layer = 0
	b.collision_mask = Kit.LAYER_WORLD | Kit.LAYER_PROP | Kit.LAYER_ENEMY | Kit.LAYER_AGENT
	var cs := CollisionShape3D.new()
	var cap := CapsuleShape3D.new()
	cap.radius = 0.45
	cap.height = 1.7
	cs.shape = cap
	cs.position = Vector3(0, 0.85, 0)
	b.add_child(cs)
	var cm := CapsuleMesh.new()
	cm.radius = 0.45
	cm.height = 1.7
	var body := Kit.mesh_instance(cm, Kit.mat(Kit.C_ENEMY))
	body.position = Vector3(0, 0.85, 0)
	b.add_child(body)
	var eye := Kit.mesh_instance(Kit.box_mesh(Vector3(0.55, 0.14, 0.14)), Kit.mat(Color(1.0, 0.85, 0.1), true))
	eye.position = Vector3(0, 1.4, -0.38)
	b.add_child(eye)
	b.set_meta("enemy_idx", i)
	add_child(b)
	b.visible = false
	b.position = Vector3(0, -40, 0)
	return {"body": b, "alive": false, "hp": 0, "mode": "static", "side": 1.0, "timer": 0, "cooldown": 0, "target": Vector3.ZERO}


# ------------------------------------------------------------------ episode
func _pick_stage() -> int:
	if params.has("stage_probs"):
		var probs: Array = params["stage_probs"]
		var r: float = rng.randf() * probs.reduce(func(a, b): return a + b, 0.0)
		for i in probs.size():
			r -= probs[i]
			if r <= 0.0:
				return i
		return probs.size() - 1
	return clampi(int(params.get("stage", 0)), 0, STAGES.size() - 1)


func _rand_in_room(margin := 1.5) -> Vector3:
	var h := ROOM / 2.0 - margin
	return Vector3(rng.randf_range(-h, h), 0.0, rng.randf_range(-h, h))


func _clear_of_obstacles(p: Vector3, pad := 1.0) -> bool:
	for o in obstacles:
		if o.position.y < -5.0:
			continue
		var s: Vector3 = (o.get_child(0) as CollisionShape3D).shape.size
		if absf(p.x - o.position.x) < s.x / 2 + pad and absf(p.z - o.position.z) < s.z / 2 + pad:
			return false
	return true


func reset_episode() -> void:
	stage = _pick_stage()
	cfg = STAGES[stage]
	max_ticks = int(cfg["time"]) * 60
	var n_obs := rng.randi_range(0, 3) if stage < 3 else rng.randi_range(2, 5)
	var order := range(obstacles.size())
	for k in range(order.size() - 1, 0, -1):  # deterministic Fisher-Yates with the arena rng
		var j := rng.randi_range(0, k)
		var tmp = order[k]
		order[k] = order[j]
		order[j] = tmp
	for k in obstacles.size():
		var o = obstacles[order[k]]
		var s: Vector3 = (o.get_child(0) as CollisionShape3D).shape.size
		if k < n_obs:
			var p := _rand_in_room(3.0)
			o.position = Vector3(p.x, s.y / 2.0, p.z)
		else:
			o.position = Vector3(0, -20, 0)
	# agent
	var ap := _rand_in_room(2.0)
	for attempt in 30:
		if _clear_of_obstacles(ap, 1.2):
			break
		ap = _rand_in_room(2.0)
	agent.place(Vector3(ap.x, 0.05, ap.z), yaw_toward(ap, Vector3.ZERO) + rng.randf_range(-0.5, 0.5))
	# enemies
	var n := rng.randi_range(cfg["n"][0], cfg["n"][1])
	for i in MAX_ENEMIES:
		var e: Dictionary = enemies[i]
		if i >= n:
			_disable_enemy(e)
			continue
		var p := Vector3.ZERO
		for attempt in 40:
			if cfg["spawn"] == "front":
				var ang: float = agent.yaw + rng.randf_range(-0.85, 0.85)
				var dist := rng.randf_range(6.0, 13.0)
				p = ap + Vector3(-sin(ang), 0, -cos(ang)) * dist
			else:
				p = _rand_in_room(1.5)
			var inside := absf(p.x) < ROOM / 2 - 1.2 and absf(p.z) < ROOM / 2 - 1.2
			if inside and p.distance_to(ap) > 5.0 and _clear_of_obstacles(p, 1.0):
				break
		p.x = clampf(p.x, -ROOM / 2 + 1.2, ROOM / 2 - 1.2)
		p.z = clampf(p.z, -ROOM / 2 + 1.2, ROOM / 2 - 1.2)
		var b: CharacterBody3D = e["body"]
		b.position = Vector3(p.x, 0.02, p.z)
		b.visible = true
		b.collision_layer = Kit.LAYER_ENEMY
		e["alive"] = true
		e["hp"] = int(cfg["hp"])
		var modes: Array = cfg["modes"]
		e["mode"] = modes[rng.randi_range(0, modes.size() - 1)]
		e["side"] = 1.0 if rng.randf() < 0.5 else -1.0
		e["timer"] = rng.randi_range(30, 120)
		e["cooldown"] = rng.randi_range(40, int(cfg.get("interval", 60)) + 40)
		e["target"] = to_global(_rand_in_room())
		b.rotation.y = yaw_toward(b.global_position, agent.global_position)
	for pr in projectiles:
		_kill_proj(pr)
	dmg_timer = 0.0
	hit_timer = 0.0
	tracer.visible = false
	stats = {"kills": 0, "shots": 0, "hits": 0, "damage": 0.0, "win": 0, "died": 0, "stage": stage, "n_enemies": n}


func _disable_enemy(e: Dictionary) -> void:
	e["alive"] = false
	var b: CharacterBody3D = e["body"]
	b.visible = false
	b.collision_layer = 0
	b.position = Vector3(0, -40, 0)


func _kill_proj(pr: Dictionary) -> void:
	pr["alive"] = false
	pr["mesh"].visible = false


# ------------------------------------------------------------------ tick
func pre_tick(dt: float) -> void:
	var speed: float = cfg["speed"]
	for e in enemies:
		if not e["alive"]:
			continue
		var b: CharacterBody3D = e["body"]
		var to_a: Vector3 = agent.global_position - b.global_position
		to_a.y = 0.0
		var dist := to_a.length()
		var dir := to_a / maxf(dist, 0.01)
		b.rotation.y = atan2(-dir.x, -dir.z)
		var v := Vector3.ZERO
		e["timer"] -= 1
		match e["mode"]:
			"strafe":
				if e["timer"] <= 0:
					e["side"] = -e["side"]
					e["timer"] = rng.randi_range(40, 120)
				v = dir.cross(Vector3.UP) * e["side"] * speed
				if dist < 5.0:
					v -= dir * speed * 0.6
				elif dist > 14.0:
					v += dir * speed * 0.6
			"wander":
				var tgt: Vector3 = e["target"]
				var d2 := tgt - b.global_position
				d2.y = 0.0
				if d2.length() < 1.0 or e["timer"] <= 0:
					e["target"] = to_global(_rand_in_room())
					e["timer"] = 240
				v = d2.normalized() * speed
			"chase":
				if dist > 8.0:
					v = dir * speed
				elif dist < 4.0:
					v = -dir * speed
				else:
					if e["timer"] <= 0:
						e["side"] = -e["side"]
						e["timer"] = rng.randi_range(40, 120)
					v = dir.cross(Vector3.UP) * e["side"] * speed * 0.7
		if speed > 0.0:
			b.velocity = v
			b.move_and_slide()
			b.position.y = 0.02
		if cfg["fire"] and agent.alive:
			e["cooldown"] -= 1
			if e["cooldown"] <= 0:
				var eye := b.global_position + Vector3(0, 1.4, 0) + dir * 0.6
				var chest: Vector3 = agent.global_position + Vector3(0, 1.0, 0)
				if ray(eye, chest, Kit.LAYER_WORLD | Kit.LAYER_PROP).is_empty():
					_fire(eye, chest)
					e["cooldown"] = int(cfg["interval"]) + rng.randi_range(-15, 15)
				else:
					e["cooldown"] = 10


func _fire(origin: Vector3, chest: Vector3) -> void:
	var ps: float = cfg["proj_speed"]
	var t_hit := origin.distance_to(chest) / ps
	var target: Vector3 = chest + agent.velocity * t_hit * float(cfg["lead"])
	var d := (target - origin).normalized()
	var nz: float = cfg["noise"]
	d = (d + Vector3(rng.randfn(0, nz), rng.randfn(0, nz * 0.5), rng.randfn(0, nz))).normalized()
	for pr in projectiles:
		if not pr["alive"]:
			pr["alive"] = true
			pr["pos"] = origin
			pr["vel"] = d * ps
			pr["ttl"] = 240
			pr["mesh"].global_position = origin
			pr["mesh"].visible = true
			return


func _hits_agent(pt: Vector3) -> bool:
	var a: Vector3 = agent.global_position
	var y := clampf(pt.y, a.y + 0.4, a.y + 1.4)
	return pt.distance_to(Vector3(a.x, y, a.z)) < 0.4 + PROJ_R


func post_tick(dt: float) -> void:
	dmg_timer = maxf(0.0, dmg_timer - dt)
	hit_timer = maxf(0.0, hit_timer - dt * 2.0)
	if tracer_ticks > 0:
		tracer_ticks -= 1
		tracer.visible = tracer_ticks > 0
	for pr in projectiles:
		if not pr["alive"]:
			continue
		var p0: Vector3 = pr["pos"]
		var p1: Vector3 = p0 + pr["vel"] * dt
		if agent.alive and (_hits_agent(p0.lerp(p1, 0.5)) or _hits_agent(p1)):
			_damage_agent(-(pr["vel"] as Vector3).normalized())
			_kill_proj(pr)
			if ended:
				return
			continue
		if not ray(p0, p1, Kit.LAYER_WORLD | Kit.LAYER_PROP).is_empty():
			_kill_proj(pr)
			continue
		pr["pos"] = p1
		pr["ttl"] -= 1
		pr["mesh"].global_position = p1
		if pr["ttl"] <= 0:
			_kill_proj(pr)


func _damage_agent(from_dir: Vector3) -> void:
	var dmg := 0.25
	agent.hp -= dmg
	agent.add_reward(-0.4 * dmg)
	stats["damage"] += dmg
	var b := local_bearing(agent, from_dir)
	dmg_dir = Vector2(sin(b), cos(b))
	dmg_timer = 1.0
	if agent.hp <= 0.001:
		agent.alive = false
		agent.add_reward(-1.0)
		stats["died"] = 1
		end_episode()


func on_primary(ag) -> void:
	stats["shots"] += 1
	ag.add_reward(-0.005)
	var from: Vector3 = ag.eye_position()
	var to: Vector3 = from + ag.aim_direction() * 60.0
	var hit := ray(from, to, Kit.LAYER_WORLD | Kit.LAYER_ENEMY | Kit.LAYER_PROP, [ag.get_rid()])
	var end_pt: Vector3 = to if hit.is_empty() else hit["position"]
	_show_tracer(from + ag.aim_direction() * 0.6 + Vector3(0, -0.15, 0), end_pt)
	if hit.is_empty() or not hit["collider"].has_meta("enemy_idx"):
		return
	var e: Dictionary = enemies[int(hit["collider"].get_meta("enemy_idx"))]
	if not e["alive"]:
		return
	e["hp"] -= 1
	stats["hits"] += 1
	hit_timer = 1.0
	ag.add_reward(0.2)
	if e["hp"] <= 0:
		_disable_enemy(e)
		stats["kills"] += 1
		ag.add_reward(1.0)
		var any_alive := false
		for o in enemies:
			any_alive = any_alive or o["alive"]
		if not any_alive:
			stats["win"] = 1
			ag.add_reward(2.0)
			end_episode()


func _show_tracer(a: Vector3, b: Vector3) -> void:
	var seg_len := a.distance_to(b)
	if seg_len < 0.1:
		return
	tracer.global_position = (a + b) / 2.0
	tracer.look_at((a + b) / 2.0 + (b - a), Vector3.UP if absf((b - a).normalized().y) < 0.99 else Vector3.RIGHT)
	tracer.scale = Vector3(1, 1, seg_len)
	tracer.visible = true
	tracer_ticks = 3


# ------------------------------------------------------------------ observation extras
func extra_vec_size() -> int:
	return 13


func extra_vec(ag) -> Array:
	var alive_n := 0
	for e in enemies:
		if e["alive"]:
			alive_n += 1
	var hearing := [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	var a: Vector3 = ag.global_position + Vector3(0, 1.0, 0)
	for pr in projectiles:
		if not pr["alive"]:
			continue
		var rel: Vector3 = pr["pos"] - a
		var d := rel.length()
		if d > 12.0 or (pr["vel"] as Vector3).dot(-rel) <= 0.0:
			continue
		var bearing := local_bearing(ag, rel)
		var bin := int(floor((bearing + PI) / (TAU / 8.0))) % 8
		hearing[bin] = maxf(hearing[bin], 1.0 - d / 12.0)
	return [dmg_dir.x * dmg_timer, dmg_dir.y * dmg_timer, dmg_timer, hit_timer,
		float(alive_n) / float(max(1, int(stats.get("n_enemies", 1))))] + hearing


func spectator_pose() -> Array:
	return [Vector3(0, 17, 19), Vector3(0, 0, 0)]


func topdown_extent() -> float:
	return ROOM + 2.0
