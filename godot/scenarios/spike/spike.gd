# Bridge spike: a plain room with N agents. Each agent carries 8 unshaded quads (red level i*32+16);
# every tick only quad (ticks % 8) is *moved* in front of the camera, the others are parked behind it.
# The same counter is sent in the vec obs, so Python can verify that every image reflects the
# transforms of the decision tick (catches stale transforms as well as stale frames).
extends "res://common/arena_base.gd"

var life_ticks := 0
var quads: Array = []
var levels: Array = []


func build() -> void:
	max_ticks = 300
	Kit.walled_room(self, 20, 20, 3, Kit.checker_mat(Kit.C_FLOOR_A, Kit.C_FLOOR_B), Kit.mat(Kit.C_WALL))
	for p in [Vector3(4, 1, 3), Vector3(-5, 1, -2), Vector3(1, 1, -6)]:
		Kit.static_box(self, Vector3(2, 2, 2), p, Kit.mat(Kit.C_OBSTACLE))
	for i in 8:
		var m := StandardMaterial3D.new()
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		m.albedo_color = Color8(i * 32 + 16, 40, 200)
		levels.append(m)
	var n := int(params.get("agents_per_arena", 4))
	var qm := QuadMesh.new()
	qm.size = Vector2(0.25, 0.25)
	for i in n:
		var ag = spawn_agent(0, "default", Kit.C_TEAM[i % 4])
		var set := []
		for k in 8:
			var q := MeshInstance3D.new()
			q.mesh = qm
			q.material_override = levels[k]
			q.position = Vector3(0, 0, 5.0)
			ag.head.add_child(q)
			set.append(q)
		quads.append(set)


func reset_episode() -> void:
	var n := agents.size()
	for i in n:
		var ang := TAU * i / n + rng.randf_range(-0.3, 0.3)
		agents[i].place(Vector3(cos(ang) * 5, 0.05, sin(ang) * 5), rng.randf_range(-PI, PI))
	_update_quads()


func post_tick(_dt: float) -> void:
	life_ticks += 1
	_update_quads()


func _update_quads() -> void:
	var k := life_ticks % 8
	for set in quads:
		for j in 8:
			set[j].position = Vector3(0, 0, -0.9) if j == k else Vector3(0, 0, 5.0)


func extra_vec_size() -> int:
	return 1


func extra_vec(_agent) -> Array:
	return [float(life_ticks % 8)]
