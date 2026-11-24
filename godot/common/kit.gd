# Procedural building helpers shared by all scenarios (no hand-written .tscn needed).
extends RefCounted

const LAYER_WORLD := 1    # static level geometry
const LAYER_AGENT := 2    # learning agents
const LAYER_ENEMY := 4    # scripted bots
const LAYER_PROP := 8     # movable boxes / ramps
const LAYER_PROJ := 16    # projectiles (not physical, kept for queries)

const C_FLOOR_A := Color(0.36, 0.38, 0.42)
const C_FLOOR_B := Color(0.30, 0.32, 0.36)
const C_WALL := Color(0.62, 0.64, 0.68)
const C_OBSTACLE := Color(0.45, 0.55, 0.62)
const C_AGENT := Color(0.15, 0.45, 0.95)
const C_TEAM := [Color(0.15, 0.45, 0.95), Color(0.98, 0.55, 0.12), Color(0.2, 0.8, 0.35), Color(0.75, 0.3, 0.9)]
const C_ENEMY := Color(0.9, 0.12, 0.12)
const C_PROJ := Color(1.0, 0.9, 0.1)
const C_BOX := Color(0.85, 0.7, 0.35)
const C_RAMP := Color(0.55, 0.4, 0.75)
const C_GOAL := Color(0.15, 0.95, 0.4)
const C_BUTTON := Color(0.95, 0.2, 0.75)

static var _mat_cache := {}


static func mat(color: Color, unshaded := false, emission := 0.0) -> StandardMaterial3D:
	var key := "%s|%s|%s" % [color.to_html(), unshaded, emission]
	if _mat_cache.has(key):
		return _mat_cache[key]
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = 0.85
	if unshaded:
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	if emission > 0.0:
		m.emission_enabled = true
		m.emission = color
		m.emission_energy_multiplier = emission
	_mat_cache[key] = m
	return m


static func checker_mat(c1: Color, c2: Color, tile_m := 2.0) -> StandardMaterial3D:
	var key := "checker|%s|%s|%s" % [c1.to_html(), c2.to_html(), tile_m]
	if _mat_cache.has(key):
		return _mat_cache[key]
	var img := Image.create(16, 16, false, Image.FORMAT_RGB8)
	for y in 16:
		for x in 16:
			img.set_pixel(x, y, c1 if ((x / 8 + y / 8) % 2 == 0) else c2)
	img.generate_mipmaps()
	var m := StandardMaterial3D.new()
	m.albedo_texture = ImageTexture.create_from_image(img)
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	m.uv1_triplanar = true
	m.uv1_world_triplanar = true
	m.uv1_scale = Vector3.ONE / (2.0 * tile_m)
	m.roughness = 1.0
	_mat_cache[key] = m
	return m


static func mesh_instance(mesh: Mesh, material: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = material
	return mi


static func box_mesh(size: Vector3) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = size
	return b


static func static_box(parent: Node, size: Vector3, pos: Vector3, material: Material, layer := LAYER_WORLD) -> StaticBody3D:
	var body := StaticBody3D.new()
	body.collision_layer = layer
	body.collision_mask = 0
	var cs := CollisionShape3D.new()
	var sh := BoxShape3D.new()
	sh.size = size
	cs.shape = sh
	body.add_child(cs)
	body.add_child(mesh_instance(box_mesh(size), material))
	parent.add_child(body)
	body.position = pos
	return body


static func static_cylinder(parent: Node, radius: float, height: float, pos: Vector3, material: Material) -> StaticBody3D:
	var body := StaticBody3D.new()
	body.collision_layer = LAYER_WORLD
	body.collision_mask = 0
	var cs := CollisionShape3D.new()
	var sh := CylinderShape3D.new()
	sh.radius = radius
	sh.height = height
	cs.shape = sh
	body.add_child(cs)
	var cm := CylinderMesh.new()
	cm.top_radius = radius
	cm.bottom_radius = radius
	cm.height = height
	cm.radial_segments = 48
	body.add_child(mesh_instance(cm, material))
	parent.add_child(body)
	body.position = pos
	return body


static func rigid_box(parent: Node, size: Vector3, pos: Vector3, material: Material, mass := 20.0) -> RigidBody3D:
	var body := RigidBody3D.new()
	body.collision_layer = LAYER_PROP
	body.collision_mask = LAYER_WORLD | LAYER_AGENT | LAYER_PROP
	body.mass = mass
	var pm := PhysicsMaterial.new()
	pm.friction = 0.9
	body.physics_material_override = pm
	var cs := CollisionShape3D.new()
	var sh := BoxShape3D.new()
	sh.size = size
	cs.shape = sh
	body.add_child(cs)
	body.add_child(mesh_instance(box_mesh(size), material))
	parent.add_child(body)
	body.position = pos
	return body


static func rigid_ramp(parent: Node, size: Vector3, pos: Vector3, material: Material, mass := 30.0) -> RigidBody3D:
	# Wedge rising along -z (PrismMesh with left_to_right=0 gives a right-angle wedge).
	var body := RigidBody3D.new()
	body.collision_layer = LAYER_PROP
	body.collision_mask = LAYER_WORLD | LAYER_AGENT | LAYER_PROP
	body.mass = mass
	var pm := PhysicsMaterial.new()
	pm.friction = 1.0
	body.physics_material_override = pm
	var prism := PrismMesh.new()
	prism.size = size
	prism.left_to_right = 0.0
	var mi := mesh_instance(prism, material)
	mi.rotation.y = PI / 2.0
	body.add_child(mi)
	var cs := CollisionShape3D.new()
	cs.shape = prism.create_convex_shape()
	cs.rotation.y = PI / 2.0
	body.add_child(cs)
	parent.add_child(body)
	body.position = pos
	return body


static func walled_room(parent: Node, size_x: float, size_z: float, wall_h: float, floor_mat: Material, wall_mat: Material) -> void:
	static_box(parent, Vector3(size_x + 2.0, 1.0, size_z + 2.0), Vector3(0, -0.5, 0), floor_mat)
	var t := 0.5
	static_box(parent, Vector3(size_x + 2 * t, wall_h, t), Vector3(0, wall_h / 2, -size_z / 2 - t / 2), wall_mat)
	static_box(parent, Vector3(size_x + 2 * t, wall_h, t), Vector3(0, wall_h / 2, size_z / 2 + t / 2), wall_mat)
	static_box(parent, Vector3(t, wall_h, size_z), Vector3(-size_x / 2 - t / 2, wall_h / 2, 0), wall_mat)
	static_box(parent, Vector3(t, wall_h, size_z), Vector3(size_x / 2 + t / 2, wall_h / 2, 0), wall_mat)


static func sphere_mesh(radius: float) -> SphereMesh:
	var s := SphereMesh.new()
	s.radius = radius
	s.height = radius * 2.0
	s.radial_segments = 16
	s.rings = 8
	return s


static func teleport_rigid(body: RigidBody3D, xform: Transform3D) -> void:
	body.global_transform = xform
	var rid := body.get_rid()
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, xform)
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3.ZERO)
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
	body.linear_velocity = Vector3.ZERO
	body.angular_velocity = Vector3.ZERO
