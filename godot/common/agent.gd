# Learning agent body: a CharacterBody3D capsule with a first-person camera rendered into
# its own SubViewport. Shared by every scenario so one network design works everywhere.
#
# Action (MultiDiscrete [3,3,5,3,2,2]):
#   0 move    : 0 none, 1 forward, 2 back
#   1 strafe  : 0 none, 1 left, 2 right
#   2 yaw     : 0 none, 1 left small, 2 right small, 3 left large, 4 right large
#   3 pitch   : 0 none, 1 up, 2 down
#   4 jump    : 0/1
#   5 primary : 0/1  (fire / shove / tag / grab depending on scenario)
extends CharacterBody3D

const Kit = preload("res://common/kit.gd")

const NVEC := [3, 3, 5, 3, 2, 2]
const CAM_FORWARD := 0.25   # camera stays inside the 0.4 m capsule (+0.05 near plane) so it never clips walls
const BASE_VEC_DIM := 8
const YAW_RATES := [0.0, 1.6, -1.6, 4.7, -4.7]  # rad/s (positive = turn left)
const MOVE_DIR := [0.0, 1.0, -1.0]
const STRAFE_DIR := [0.0, -1.0, 1.0]
const PITCH_DIR := [0.0, 1.0, -1.0]

var move_speed := 6.0
var jump_speed := 7.0
var gravity := 20.0
var pitch_rate := 1.6
var pitch_limit := deg_to_rad(60.0)
var cooldown_ticks := 15
var push_impulse := 1.5

# identity
var arena = null
var index := -1           # index within this Godot process
var team := 0
var policy_name := "default"
var color := Kit.C_AGENT

# game state
var yaw := 0.0
var pitch := 0.0
var hp := 1.0
var alive := true
var frozen := false        # cannot act (e.g. hide&seek prep phase for seekers)
var blind := false         # observation image is zeroed
var cooldown := 0
var walk_vel := Vector3.ZERO
var knockback := Vector3.ZERO
var primary_pressed := false  # set for one tick when primary fires

# RL bookkeeping
var action := PackedInt32Array([0, 0, 0, 0, 0, 0])
var hold_still := false
var reward := 0.0
var ep_return := 0.0
var done := false
var truncated := false
var info = null

var head: Node3D
var cam_vp: SubViewport
var cam: Camera3D
var body_mesh: MeshInstance3D
var visor: MeshInstance3D


var self_layer := 2         # render layer bit used for this agent's own meshes (hidden from own camera)


func setup(p_color: Color, pov_size: Vector2i, slot := 0, fov := 90.0, far := 70.0) -> void:
	color = p_color
	self_layer = 1 << (1 + slot)
	collision_layer = Kit.LAYER_AGENT
	collision_mask = Kit.LAYER_WORLD | Kit.LAYER_AGENT | Kit.LAYER_PROP | Kit.LAYER_ENEMY
	floor_snap_length = 0.3
	floor_max_angle = deg_to_rad(50.0)
	var cs := CollisionShape3D.new()
	var cap := CapsuleShape3D.new()
	cap.radius = 0.4
	cap.height = 1.8
	cs.shape = cap
	cs.position = Vector3(0, 0.9, 0)
	add_child(cs)
	var cm := CapsuleMesh.new()
	cm.radius = 0.4
	cm.height = 1.8
	body_mesh = Kit.mesh_instance(cm, Kit.mat(color))
	body_mesh.position = Vector3(0, 0.9, 0)
	body_mesh.layers = self_layer
	add_child(body_mesh)
	head = Node3D.new()
	head.position = Vector3(0, 1.55, 0)
	add_child(head)
	visor = Kit.mesh_instance(Kit.box_mesh(Vector3(0.5, 0.16, 0.12)), Kit.mat(Color(0.05, 0.05, 0.08)))
	visor.position = Vector3(0, 0.02, -0.36)
	visor.layers = self_layer
	head.add_child(visor)
	cam_vp = SubViewport.new()
	cam_vp.size = pov_size
	cam_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	cam_vp.msaa_3d = Viewport.MSAA_DISABLED
	cam_vp.positional_shadow_atlas_size = 0
	add_child(cam_vp)
	cam = Camera3D.new()
	cam.fov = fov
	cam.near = 0.05
	cam.far = far
	cam.cull_mask = 0xFFFFF & ~self_layer
	cam_vp.add_child(cam)
	cam.current = true


func set_color(c: Color) -> void:
	color = c
	body_mesh.material_override = Kit.mat(c)


func add_self_mesh(mi: VisualInstance3D) -> void:
	# extra cosmetic meshes (e.g. gun) that the agent's own camera should not see
	mi.layers = self_layer


func place(pos: Vector3, p_yaw: float) -> void:
	position = pos
	yaw = p_yaw
	pitch = 0.0
	rotation = Vector3(0, yaw, 0)
	head.rotation = Vector3.ZERO
	velocity = Vector3.ZERO
	walk_vel = Vector3.ZERO
	knockback = Vector3.ZERO
	cooldown = 0
	hp = 1.0
	alive = true
	frozen = false
	blind = false
	visible = true
	collision_layer = Kit.LAYER_AGENT
	sync_camera()


func forward() -> Vector3:
	return -global_transform.basis.z


func eye_position() -> Vector3:
	return head.global_position


func aim_direction() -> Vector3:
	return -head.global_transform.basis.z


func physics_step(dt: float) -> void:
	primary_pressed = false
	var a := action
	var idle := hold_still or frozen or not alive
	if not idle:
		yaw += YAW_RATES[a[2]] * dt
		pitch = clampf(pitch + PITCH_DIR[a[3]] * pitch_rate * dt, -pitch_limit, pitch_limit)
	rotation.y = yaw
	head.rotation.x = pitch
	var target := Vector3.ZERO
	if not idle:
		var local := Vector3(STRAFE_DIR[a[1]], 0.0, -MOVE_DIR[a[0]])
		if local.length_squared() > 0.0:
			target = (transform.basis * local).normalized() * move_speed
	var accel := 50.0 if is_on_floor() else 12.0
	walk_vel = walk_vel.move_toward(target, accel * dt)
	knockback = knockback.move_toward(Vector3.ZERO, (14.0 if is_on_floor() else 2.0) * dt)
	velocity.x = walk_vel.x + knockback.x
	velocity.z = walk_vel.z + knockback.z
	if is_on_floor():
		if not idle and a[4] == 1:
			velocity.y = jump_speed
	velocity.y -= gravity * dt
	move_and_slide()
	# push movable props we walk into (CharacterBody3D does not push RigidBody3D by itself)
	for i in get_slide_collision_count():
		var c := get_slide_collision(i)
		var rb := c.get_collider() as RigidBody3D
		if rb != null:
			var n := c.get_normal()
			n.y = 0.0
			rb.apply_central_impulse(-n * push_impulse)
	if cooldown > 0:
		cooldown -= 1
	if not idle and a[5] == 1 and cooldown == 0:
		cooldown = cooldown_ticks
		primary_pressed = true
		arena.on_primary(self)
	sync_camera()


func sync_camera() -> void:
	if head == null:
		return
	cam.global_transform = head.global_transform.translated_local(Vector3(0, 0, -CAM_FORWARD))


func add_reward(r: float) -> void:
	reward += r
	ep_return += r


func base_vec() -> Array:
	var lv := global_transform.basis.inverse() * velocity
	return [
		lv.x / 8.0, lv.y / 8.0, lv.z / 8.0,
		1.0 if is_on_floor() else 0.0,
		hp,
		float(cooldown) / float(cooldown_ticks),
		pitch / pitch_limit,
		arena.time_left_frac(),
	]


func pov_bytes() -> PackedByteArray:
	var n := cam_vp.size.x * cam_vp.size.y * 3
	if blind:
		var z := PackedByteArray()
		z.resize(n)
		return z
	var img := cam_vp.get_texture().get_image()
	if img.get_format() != Image.FORMAT_RGB8:
		img.convert(Image.FORMAT_RGB8)
	return img.get_data()
