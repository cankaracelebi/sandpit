# Pooled GPU particle effects + flash lights for cinema mode. Particles advance by the frame step
# passed to RenderingServer.force_draw, so offline rendering stays deterministic.
extends Node3D

var pools := {}
var lights: Array = []
var light_i := 0


func setup() -> void:
	pools["muzzle"] = _pool(6, Color(1.0, 0.85, 0.35), 18, 0.14, 5.0, 0.035, 0.0, 25.0, 6.0)
	pools["impact"] = _pool(8, Color(1.0, 0.7, 0.25), 22, 0.35, 4.0, 0.03, 9.8, 70.0, 5.0)
	pools["spark_red"] = _pool(6, Color(1.0, 0.25, 0.15), 40, 0.6, 5.5, 0.05, 9.8, 180.0, 5.0)
	pools["explode"] = _pool(4, Color(1.0, 0.45, 0.1), 70, 0.8, 6.0, 0.07, 6.0, 180.0, 4.0)
	pools["dust"] = _pool(8, Color(0.75, 0.72, 0.66), 24, 0.6, 2.2, 0.09, 1.0, 80.0, 0.0)
	pools["confetti"] = _pool(3, Color(0.3, 1.0, 0.5), 90, 1.6, 7.0, 0.06, 6.0, 40.0, 3.0)
	pools["confetti2"] = _pool(3, Color(1.0, 0.85, 0.2), 90, 1.6, 7.0, 0.06, 6.0, 40.0, 3.0)
	pools["confetti3"] = _pool(3, Color(0.4, 0.6, 1.0), 90, 1.6, 7.0, 0.06, 6.0, 40.0, 3.0)
	for i in 6:
		var l := OmniLight3D.new()
		l.omni_range = 5.0
		l.light_energy = 0.0
		l.shadow_enabled = false
		add_child(l)
		lights.append({"light": l, "frames": 0, "energy": 0.0})


func _pool(n: int, color: Color, amount: int, lifetime: float, speed: float, size: float, gravity: float,
		spread: float, emission: float) -> Dictionary:
	var items := []
	var pm := ParticleProcessMaterial.new()
	pm.direction = Vector3(0, 1, 0)
	pm.spread = spread
	pm.initial_velocity_min = speed * 0.35
	pm.initial_velocity_max = speed
	pm.gravity = Vector3(0, -gravity, 0)
	pm.damping_min = 1.0
	pm.damping_max = 3.0
	pm.scale_min = 0.6
	pm.scale_max = 1.3
	var curve := Curve.new()
	curve.add_point(Vector2(0, 1))
	curve.add_point(Vector2(1, 0))
	var ct := CurveTexture.new()
	ct.curve = curve
	pm.scale_curve = ct
	var mesh := SphereMesh.new()
	mesh.radius = size
	mesh.height = size * 2.0
	mesh.radial_segments = 6
	mesh.rings = 3
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	if emission > 0.0:
		mat.emission_enabled = true
		mat.emission = color
		mat.emission_energy_multiplier = emission
	mat.roughness = 1.0
	mesh.material = mat
	for i in n:
		var p := GPUParticles3D.new()
		p.one_shot = true
		p.emitting = false
		p.amount = amount
		p.lifetime = lifetime
		p.explosiveness = 0.95
		p.local_coords = false
		p.fixed_fps = 30
		p.interpolate = false
		p.process_material = pm
		p.draw_pass_1 = mesh
		p.visibility_aabb = AABB(Vector3(-8, -8, -8), Vector3(16, 16, 16))
		add_child(p)
		items.append(p)
	return {"items": items, "i": 0}


func emit(kind: String, pos: Vector3, dir := Vector3.ZERO) -> void:
	var pool: Dictionary = pools[kind]
	var p: GPUParticles3D = pool["items"][pool["i"]]
	pool["i"] = (pool["i"] + 1) % pool["items"].size()
	p.global_position = pos
	if dir.length() > 0.01:
		var up := Vector3.UP if absf(dir.normalized().dot(Vector3.UP)) < 0.95 else Vector3.RIGHT
		# particle "direction" is +Y in emitter space: point the emitter's +Y along dir
		p.global_transform = Transform3D(Basis.looking_at(dir, up) * Basis(Vector3.RIGHT, -PI / 2.0), pos)
	else:
		p.global_transform = Transform3D(Basis(), pos)
	p.restart()
	p.emitting = true


func flash(pos: Vector3, color: Color, energy := 3.0, frames := 2) -> void:
	var l: Dictionary = lights[light_i]
	light_i = (light_i + 1) % lights.size()
	var o: OmniLight3D = l["light"]
	o.global_position = pos
	o.light_color = color
	o.light_energy = energy
	l["frames"] = frames
	l["energy"] = energy


func step() -> void:
	for l in lights:
		if l["frames"] > 0:
			l["frames"] -= 1
			if l["frames"] == 0:
				l["light"].light_energy = 0.0


func trail(parent: Node3D, color: Color) -> GPUParticles3D:
	var p := GPUParticles3D.new()
	p.amount = 24
	p.lifetime = 0.25
	p.local_coords = false
	p.fixed_fps = 30
	var pm := ParticleProcessMaterial.new()
	pm.spread = 10.0
	pm.initial_velocity_min = 0.0
	pm.initial_velocity_max = 0.3
	pm.gravity = Vector3.ZERO
	var curve := Curve.new()
	curve.add_point(Vector2(0, 1))
	curve.add_point(Vector2(1, 0))
	var ct := CurveTexture.new()
	ct.curve = curve
	pm.scale_curve = ct
	p.process_material = pm
	var mesh := SphereMesh.new()
	mesh.radius = 0.13
	mesh.height = 0.26
	mesh.radial_segments = 6
	mesh.rings = 3
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.emission_enabled = true
	mat.emission = color
	mat.emission_energy_multiplier = 3.0
	mesh.material = mat
	p.draw_pass_1 = mesh
	p.visibility_aabb = AABB(Vector3(-6, -6, -6), Vector3(12, 12, 12))
	parent.add_child(p)
	return p
