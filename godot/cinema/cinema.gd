# Cinema mode: offline high-quality re-rendering of a recorded replay.
#   godot --path godot --rendering-method forward_plus -- --mode=cinema --replay=<file> --out_dir=<dir>
# Rebuilds the scenario arena from the same script + params, then for every replay frame applies
# the recorded transforms and renders observer + chase cameras at 1080p. Visual upgrades are all
# here and never touch training: Forward+, an HDRI sky, soft PCSS shadows, SSAO/SSIL, glow, Kenney
# CC0 animated characters, prototype-texture materials and particle FX.
extends Node3D

const Kit = preload("res://common/kit.gd")
const Rec = preload("res://cinema/replay_recorder.gd")
const FX = preload("res://cinema/fx.gd")
const SCENARIOS := {
	"spike": "res://scenarios/spike/spike.gd",
	"shooter": "res://scenarios/shooter/shooter.gd",
	"sumo": "res://scenarios/sumo/sumo.gd",
	"tag": "res://scenarios/hideseek/hideseek.gd",
	"hideseek": "res://scenarios/hideseek/hideseek.gd",
	"parkour": "res://scenarios/parkour/parkour.gd",
}
const CHAR_DIR := "res://assets/kenney_mini/"
const PROTO := "res://assets/kenney_proto/"
const CHAR_SCALE := 2.35
const LOOPING := ["idle", "walk", "sprint", "fall", "holding-right", "holding-both", "sit"]

var main
var args := {}
var header: Dictionary = {}
var frames: Array = []
var arena
var nodes: Array = []
var chars: Array = []
var cams: Array = []          # [name, SubViewport, Camera3D]
var out_dir := ""
var frame_i := 0
var warm := 0
var pending := false
var fx
var dt := 1.0 / 30.0
var max_frames := 1000000
var prev_vis := []
var prev_hp := []
var prev_floor := []
var proj_nodes := {}          # node idx -> trail particles
var tracer_idx := -1
var goal_idx := -1
var res := Vector2i(1920, 1080)
var orbit := 0.0
var chase_eye := Vector3.ZERO
var chase_look := Vector3.ZERO
var obs_focus := Vector3.ZERO
var mat_cache := {}
var celebrated := false
var tracer_new := false
var tracer_end := Vector3.ZERO


func start(p_main, p_args: Dictionary) -> void:
	main = p_main
	args = p_args
	out_dir = str(args.get("out_dir", "user://cinema"))
	DirAccess.make_dir_recursive_absolute(out_dir)
	_load_replay(str(args["replay"]))
	dt = float(header.get("every", 2)) / 60.0
	max_frames = int(args.get("max_frames", 1000000))
	var r := str(args.get("res", "1920x1080")).split("x")
	res = Vector2i(int(r[0]), int(r[1]))
	RenderingServer.render_loop_enabled = false
	OS.low_processor_usage_mode_sleep_usec = 0
	get_tree().root.disable_3d = true
	if int(args.get("minimize", 1)) == 1:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MINIMIZED)
	PhysicsServer3D.set_active(false)
	_build_env()
	arena = load(SCENARIOS[str(header["scenario"])]).new()
	add_child(arena)
	var params: Dictionary = header.get("params", {})
	arena.setup(0, int(header.get("seed", 0)), params, Vector2i(64, 64), true)
	nodes = Rec.tracked_nodes(arena)
	if nodes.size() != int(header["n_nodes"]):
		push_error("cinema: node count mismatch %d vs %d" % [nodes.size(), int(header["n_nodes"])])
	for a in arena.agents:
		a.cam_vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	for n in arena.find_children("*", "RigidBody3D", true, false):
		n.freeze = true
	for n in arena.find_children("*", "AnimatableBody3D", true, false):
		n.sync_to_physics = false
	fx = FX.new()
	add_child(fx)
	fx.setup()
	_beautify()
	if str(header["scenario"]) in ["shooter", "tag", "hideseek", "spike"]:
		var ground := MeshInstance3D.new()
		var pm := PlaneMesh.new()
		pm.size = Vector2(500, 500)
		ground.mesh = pm
		ground.material_override = _proto("Green", "texture_08", 8.0, Color(0.5, 0.62, 0.48))
		ground.position = Vector3(0, -0.03, 0)
		add_child(ground)
	_make_characters()
	_make_cameras()
	prev_vis.resize(nodes.size())
	prev_vis.fill(false)
	prev_hp.resize(arena.agents.size())
	prev_hp.fill(1.0)
	prev_floor.resize(arena.agents.size())
	prev_floor.fill(true)
	print("CINEMA_START frames=", frames.size(), " nodes=", nodes.size(), " res=", res)


func _load_replay(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ)
	header = JSON.parse_string(f.get_line())
	var rest := f.get_buffer(f.get_length() - f.get_position())
	var data := rest.to_float32_array()
	var F := 1 + int(header["n_nodes"]) * int(header["node_floats"]) + int(header["n_agents"]) * int(header["agent_floats"])
	var n := data.size() / F
	for i in n:
		frames.append(data.slice(i * F, (i + 1) * F))


# ------------------------------------------------------------------ look
func _build_env() -> void:
	var env := Environment.new()
	var sky := Sky.new()
	var pan := PanoramaSkyMaterial.new()
	pan.panorama = load("res://assets/hdri/kloofendal_48d_partly_cloudy_puresky_2k.hdr")
	pan.energy_multiplier = 1.0
	sky.sky_material = pan
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.sky_rotation = Vector3(0, deg_to_rad(float(args.get("sky_rot", 140.0))), 0)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 1.0
	env.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.tonemap_exposure = 1.0
	env.ssao_enabled = true
	env.ssao_radius = 1.2
	env.ssao_intensity = 1.6
	env.ssil_enabled = true
	env.ssil_radius = 4.0
	env.glow_enabled = true
	env.glow_intensity = 0.7
	env.glow_bloom = 0.04
	env.glow_hdr_threshold = 1.1
	env.fog_enabled = true
	env.fog_light_color = Color(0.72, 0.8, 0.92)
	env.fog_density = 0.0025
	env.fog_sky_affect = 0.0
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-52, -38, 0)
	sun.light_energy = 1.5
	sun.light_color = Color(1.0, 0.96, 0.9)
	sun.light_angular_distance = 1.2
	sun.shadow_enabled = true
	sun.shadow_blur = 1.2
	sun.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	sun.directional_shadow_max_distance = 80.0
	add_child(sun)
	RenderingServer.directional_shadow_atlas_set_size(8192, true)
	RenderingServer.directional_soft_shadow_filter_set_quality(RenderingServer.SHADOW_QUALITY_SOFT_HIGH)


func _proto(color_name: String, tex := "texture_02", tile_m := 2.0, tint := Color.WHITE) -> StandardMaterial3D:
	var key := "%s|%s|%s|%s" % [color_name, tex, tile_m, tint.to_html()]
	if mat_cache.has(key):
		return mat_cache[key]
	var m := StandardMaterial3D.new()
	m.albedo_texture = load(PROTO + color_name + "/" + tex + ".png")
	m.albedo_color = tint
	m.uv1_triplanar = true
	m.uv1_world_triplanar = true
	m.uv1_scale = Vector3.ONE / tile_m
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	m.roughness = 0.7
	mat_cache[key] = m
	return m


func _pick_palette(c: Color) -> String:
	var mx := maxf(c.r, maxf(c.g, c.b))
	var mn := minf(c.r, minf(c.g, c.b))
	var sat := (mx - mn) / maxf(mx, 0.001)
	if sat < 0.28:
		return "Light" if c.get_luminance() > 0.5 else "Dark"
	var h := c.h * 360.0
	if h < 15.0 or h >= 330.0:
		return "Red"
	if h < 55.0:
		return "Orange"
	if h < 170.0:
		return "Green"
	if h < 255.0:
		return "Light"   # bluish-grey surfaces read best as light grey
	return "Purple"


func _beautify() -> void:
	var skip := {}
	for a in arena.agents:
		skip[a] = true
	if "enemies" in arena:
		for e in arena.enemies:
			skip[e["body"]] = true
	for mi in arena.find_children("*", "MeshInstance3D", true, false):
		var owner_skip := false
		var p: Node = mi
		while p != null and p != arena:
			if skip.has(p):
				owner_skip = true
				break
			p = p.get_parent()
		if owner_skip:
			continue
		var m: StandardMaterial3D = mi.material_override as StandardMaterial3D
		if m == null:
			continue
		if m.shading_mode == BaseMaterial3D.SHADING_MODE_UNSHADED:
			# projectiles / tracer / button: make them glow
			var g := StandardMaterial3D.new()
			g.albedo_color = m.albedo_color
			g.emission_enabled = true
			g.emission = m.albedo_color
			g.emission_energy_multiplier = 5.0
			mi.material_override = g
			continue
		var base: Color = m.albedo_color
		var checker := m.albedo_texture != null
		if checker:
			base = (m.albedo_texture as Texture2D).get_image().get_pixel(0, 0)
		var pal := _pick_palette(base)
		var size: Vector3 = mi.get_aabb().size
		if mi.global_position.y < -10.0 and size.x * size.z > 2000.0:
			mi.material_override = _proto("Green", "texture_08", 8.0, Color(0.5, 0.62, 0.48))  # distant ground
			continue
		var is_floor := size.y < 1.2 and size.x * size.z > 60.0 and (pal == "Light" or pal == "Dark")
		if is_floor:
			mi.material_override = _proto("Dark", "texture_02", 3.0) if base.get_luminance() < 0.5 else _proto("Light", "texture_02", 3.0)
		elif pal == "Green" and base.get_luminance() < 0.35:
			mi.material_override = _proto("Green", "texture_08", 6.0, Color(0.55, 0.65, 0.55))  # distant ground
		else:
			mi.material_override = _proto(pal, "texture_02" if not checker else "texture_08", 2.0 if not checker else 1.0)


# ------------------------------------------------------------------ characters
func _model_for(team: int, idx: int) -> String:
	match str(header["scenario"]):
		"shooter":
			return "character-soldier"
		"sumo":
			return "character-knight" if team == 0 else "character-raider"
		"tag", "hideseek":
			if team == 0:
				return ["character-female-a", "character-male-a"][idx % 2]
			return ["character-zombie", "character-robot"][idx % 2]
		"parkour":
			return "character-alien"
	return "character-male-a"


func _team_color(team: int, enemy := false) -> Color:
	if enemy:
		return Color(1.0, 0.18, 0.12)
	return [Color(0.2, 0.55, 1.0), Color(1.0, 0.55, 0.1), Color(0.3, 0.9, 0.4), Color(0.8, 0.35, 1.0)][team % 4]


func _spawn_char(model: String, ring_color: Color, gun: bool) -> Dictionary:
	var root := Node3D.new()
	add_child(root)
	var inst: Node3D = (load(CHAR_DIR + model + ".glb") as PackedScene).instantiate()
	inst.scale = Vector3.ONE * CHAR_SCALE
	inst.rotation.y = PI
	root.add_child(inst)
	var ap: AnimationPlayer = inst.find_child("AnimationPlayer", true, false)
	ap.callback_mode_process = AnimationMixer.ANIMATION_CALLBACK_MODE_PROCESS_MANUAL
	for an in LOOPING:
		if ap.has_animation(an):
			ap.get_animation(an).loop_mode = Animation.LOOP_LINEAR
	ap.play("idle")
	var sk: Skeleton3D = inst.find_child("Skeleton3D", true, false)
	var ring := MeshInstance3D.new()
	var tm := TorusMesh.new()
	tm.inner_radius = 0.55
	tm.outer_radius = 0.68
	tm.rings = 32
	tm.ring_segments = 8
	ring.mesh = tm
	var rm := StandardMaterial3D.new()
	rm.albedo_color = ring_color
	rm.emission_enabled = true
	rm.emission = ring_color
	rm.emission_energy_multiplier = 2.5
	ring.material_override = rm
	ring.scale = Vector3(1, 0.25, 1)
	ring.position = Vector3(0, 0.04, 0)
	root.add_child(ring)
	var c := {"root": root, "ap": ap, "skel": sk, "arm": sk.find_bone("arm-right"), "cur": "idle",
		"action": "", "action_t": 0.0, "alive": true, "dead_t": 0.0, "recoil": 0.0, "ring": rm,
		"prev_pos": Vector3.ZERO, "gun": null}
	if gun:
		var ba := BoneAttachment3D.new()
		ba.bone_name = "arm-right"
		sk.add_child(ba)
		var g: Node3D = (load(CHAR_DIR + "weapon-shooter.glb") as PackedScene).instantiate()
		g.position = Vector3(-0.2, 0.0, 0.0)       # tuned with tools/pose_test.gd: in the right hand,
		g.rotation_degrees = Vector3(0, -90, 0)    # barrel along the forward-held arm
		ba.add_child(g)
		c["gun"] = g
		c["hold_q"] = _arm_track(ap, "holding-right", 0.0)
		c["shoot_anim"] = ap.get_animation("holding-right-shoot")
	return c


func _arm_track(ap: AnimationPlayer, anim_name: String, t: float) -> Quaternion:
	var an := ap.get_animation(anim_name)
	for i in an.get_track_count():
		if an.track_get_type(i) == Animation.TYPE_ROTATION_3D and str(an.track_get_path(i)).ends_with(":arm-right"):
			return an.rotation_track_interpolate(i, t)
	return Quaternion.IDENTITY


func _hide_meshes(n: Node) -> void:
	for mi in n.find_children("*", "MeshInstance3D", true, false):
		mi.visible = false


func _make_characters() -> void:
	for i in arena.agents.size():
		var a = arena.agents[i]
		_hide_meshes(a)
		var c := _spawn_char(_model_for(a.team, i), _team_color(a.team), str(header["scenario"]) == "shooter")
		c["kind"] = "agent"
		c["rec"] = i
		c["node"] = nodes.find(a)
		chars.append(c)
	if "enemies" in arena:
		for e in arena.enemies:
			var b: Node3D = e["body"]
			_hide_meshes(b)
			var c := _spawn_char("character-robot", _team_color(0, true), false)
			c["kind"] = "enemy"
			c["node"] = nodes.find(b)
			c["alive"] = false
			c["root"].visible = false
			chars.append(c)
	if "projectiles" in arena:
		for pr in arena.projectiles:
			var idx := nodes.find(pr["mesh"])
			proj_nodes[idx] = fx.trail(pr["mesh"], Color(1.0, 0.75, 0.2))
			var l := OmniLight3D.new()
			l.light_color = Color(1.0, 0.7, 0.3)
			l.light_energy = 1.2
			l.omni_range = 3.0
			pr["mesh"].add_child(l)
	if "tracer" in arena:
		tracer_idx = nodes.find(arena.tracer)
	if "goal_plat" in arena:
		goal_idx = nodes.find(arena.goal_plat)


# ------------------------------------------------------------------ cameras
func _make_cameras() -> void:
	var want := str(args.get("cams", "observer,chase")).split(",")
	for name in want:
		var vp := SubViewport.new()
		vp.size = res
		vp.msaa_3d = Viewport.MSAA_4X
		vp.screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA
		vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		vp.positional_shadow_atlas_size = 4096
		add_child(vp)
		var cam := Camera3D.new()
		cam.fov = 42.0 if name == "observer" else 62.0
		cam.far = 400.0
		cam.near = 0.1
		vp.add_child(cam)
		cam.current = true
		cams.append([name, vp, cam])


func _agent_pos(i: int) -> Vector3:
	return arena.agents[i].global_position


func _update_cams(snap: bool) -> void:
	var scen := str(header["scenario"])
	var focus := int(args.get("focus", 0))
	var ap := _agent_pos(focus)
	var k := 1.0 if snap else 0.08
	orbit += dt * deg_to_rad(float(args.get("orbit_speed", 5.0)))
	for entry in cams:
		var cam: Camera3D = entry[2]
		match entry[0]:
			"observer":
				var center := Vector3.ZERO
				var radius := 17.0
				var height := 21.0
				match scen:
					"sumo":
						radius = 11.0
						height = 11.5
					"tag", "hideseek":
						radius = 16.0
						height = 21.0
					"parkour":
						center = Vector3(0, ap.y, ap.z - 5.0)
						radius = 15.0
						height = 13.0
				if scen == "parkour":
					obs_focus = obs_focus.lerp(center, k)
					var off := Vector3(cos(orbit * 0.5 + 0.6), 0, sin(orbit * 0.5 + 0.6) * 0.4 + 0.5).normalized() * radius
					cam.look_at_from_position(obs_focus + off + Vector3(0, height, 0), obs_focus)
				else:
					var eye := center + Vector3(cos(orbit), 0, sin(orbit)) * radius + Vector3(0, height, 0)
					cam.look_at_from_position(eye, center + Vector3(0, -1.0, 0))
			"chase":
				var a = arena.agents[focus]
				var back: Vector3 = a.global_transform.basis.z.normalized()
				var eye_t: Vector3 = ap + back * 4.2 + Vector3(0, 2.6, 0)
				var look_t: Vector3 = ap + Vector3(0, 1.3, 0) - back * 5.0
				if snap or chase_eye.distance_to(eye_t) > 12.0:
					chase_eye = eye_t
					chase_look = look_t
				chase_eye = chase_eye.lerp(eye_t, 0.18)
				chase_look = chase_look.lerp(look_t, 0.25)
				cam.look_at_from_position(chase_eye, chase_look)
			"top":
				cam.projection = Camera3D.PROJECTION_ORTHOGONAL
				cam.size = arena.topdown_extent()
				cam.look_at_from_position(Vector3(0, 80, 0.01), Vector3.ZERO)


# ------------------------------------------------------------------ replay application
func _apply(fi: int) -> void:
	var fr: PackedFloat32Array = frames[fi]
	var nf := int(header["node_floats"])
	var o := 1
	for n in nodes.size():
		var node: Node3D = nodes[n]
		var q := Quaternion(fr[o + 3], fr[o + 4], fr[o + 5], fr[o + 6])
		var s := Vector3(fr[o + 7], fr[o + 8], fr[o + 9])
		node.transform = Transform3D(Basis(q) * Basis.from_scale(s), Vector3(fr[o], fr[o + 1], fr[o + 2]))
		var vis := fr[o + 10] > 0.5
		var was: bool = prev_vis[n]
		if n == tracer_idx and vis:
			var bz: Vector3 = node.global_transform.basis.z
			tracer_end = node.global_position - bz * 0.5
			tracer_new = not was
		if proj_nodes.has(n):
			proj_nodes[n].emitting = vis
			if was and not vis:
				fx.emit("impact", node.global_position)
		node.visible = vis
		prev_vis[n] = vis
		o += nf


func _agent_rec(fi: int, i: int) -> Array:
	var fr: PackedFloat32Array = frames[fi]
	var o := 1 + int(header["n_nodes"]) * int(header["node_floats"]) + i * int(header["agent_floats"])
	return [Vector3(fr[o], fr[o + 1], fr[o + 2]), fr[o + 3] > 0.5, fr[o + 4], fr[o + 5], int(fr[o + 6])]


func _set_anim(c: Dictionary, name: String, speed := 1.0) -> void:
	var ap: AnimationPlayer = c["ap"]
	if c["cur"] != name:
		ap.play(name, 0.12)
		c["cur"] = name
	ap.speed_scale = speed


func _update_chars(fi: int) -> void:
	var scen := str(header["scenario"])
	for c in chars:
		var node: Node3D = nodes[c["node"]]
		var vel := Vector3.ZERO
		var on_floor := true
		var alive := true
		var flags := 0
		var pitch := 0.0
		if c["kind"] == "agent":
			var r := _agent_rec(fi, c["rec"])
			vel = r[0]
			on_floor = r[1]
			pitch = r[2]
			var hp: float = r[3]
			flags = r[4]
			alive = (flags & 1) != 0
			var i: int = c["rec"]
			if hp < prev_hp[i] - 0.01:
				fx.emit("spark_red", node.global_position + Vector3(0, 1.2, 0))
			prev_hp[i] = hp
			if on_floor and not prev_floor[i] and vel.y < -3.0:
				fx.emit("dust", node.global_position + Vector3(0, 0.1, 0))
			prev_floor[i] = on_floor
			if (flags & 4) != 0:
				match scen:
					"shooter":
						c["recoil"] = 0.12
					"sumo":
						c["action"] = "attack-melee-right"
						c["action_t"] = 0.42
						fx.emit("dust", node.global_position + node.global_transform.basis.z * -0.9 + Vector3(0, 1.0, 0))
					"tag":
						c["action"] = "interact-right"
						c["action_t"] = 0.5
					_:
						c["action"] = "pick-up"
						c["action_t"] = 0.5
		else:
			alive = node.visible
			vel = (node.global_position - c["prev_pos"]) / dt if c["alive"] else Vector3.ZERO
		c["prev_pos"] = node.global_position
		var root: Node3D = c["root"]
		if alive:
			if not c["alive"]:
				root.visible = true
				c["cur"] = ""
			root.global_transform = Transform3D(Basis(Vector3.UP, node.global_rotation.y), node.global_position)
			c["dead_t"] = 0.0
		else:
			if c["alive"]:
				if c["kind"] == "enemy":
					fx.emit("explode", root.global_position + Vector3(0, 1.0, 0))
					fx.flash(root.global_position + Vector3(0, 1.2, 0), Color(1, 0.5, 0.2), 6.0, 3)
				_set_anim(c, "die")
			c["dead_t"] += dt
			if c["dead_t"] > 1.0:
				root.visible = false
		c["alive"] = alive
		if alive:
			var hs := Vector2(vel.x, vel.z).length()
			var name := "idle"
			var spd := 1.0
			if (flags & 16) != 0:      # tagged (tag mode)
				name = "sit"
			elif not on_floor and absf(vel.y) > 1.0:
				name = "jump" if vel.y > 0.0 else "fall"
			elif hs > 4.2:
				name = "sprint"
				spd = hs / 6.0
			elif hs > 0.5:
				name = "walk"
				spd = clampf(hs / 3.0, 0.6, 1.6)
			if c["action_t"] > 0.0 and name != "sit":
				name = c["action"]
				spd = 1.0
				c["action_t"] -= dt
			_set_anim(c, name, spd)
		var ap: AnimationPlayer = c["ap"]
		ap.advance(dt)
		if c["gun"] != null and alive:
			# arm pose from the pack's own "holding-right" / "holding-right-shoot" clips (rig is T-pose)
			var rec: float = c["recoil"]
			c["recoil"] = maxf(0.0, rec - dt)
			var q: Quaternion = c["hold_q"]
			if rec > 0.0:
				var sa: Animation = c["shoot_anim"]
				q = _arm_track(ap, "holding-right-shoot", (1.0 - rec / 0.12) * sa.length)
			var sk: Skeleton3D = c["skel"]
			sk.set_bone_pose_rotation(c["arm"], q)
	_fix_tracer()
	# celebration when the parkour goal is reached
	if goal_idx >= 0 and chars.size() > 0:
		var g: Node3D = nodes[goal_idx]
		var a = arena.agents[0]
		var near: bool = a.global_position.distance_to(g.global_position + Vector3(0, 0.3, 0)) < 2.4 and prev_floor[0]
		if near and not celebrated:
			for k in ["confetti", "confetti2", "confetti3"]:
				fx.emit(k, g.global_position + Vector3(0, 0.5, 0))
			chars[0]["action"] = "emote-yes"
			chars[0]["action_t"] = 1.3
		celebrated = near


func _fix_tracer() -> void:
	if tracer_idx < 0 or not nodes[tracer_idx].visible or chars.is_empty() or chars[0]["gun"] == null:
		return
	var c: Dictionary = chars[0]
	var fwd: Vector3 = -(c["root"] as Node3D).global_transform.basis.z
	var muzzle: Vector3 = (c["gun"] as Node3D).global_position + fwd * 0.55 + Vector3(0, 0.08, 0)
	var tr: Node3D = nodes[tracer_idx]
	var seg := tracer_end - muzzle
	if seg.length() < 0.2:
		return
	tr.global_transform = Transform3D(Basis.looking_at(seg, Vector3.UP if absf(seg.normalized().y) < 0.99 else Vector3.RIGHT)
		* Basis.from_scale(Vector3(1, 1, seg.length())), muzzle + seg * 0.5)
	if tracer_new:
		fx.emit("muzzle", muzzle, fwd)
		fx.emit("impact", tracer_end, -seg)
		fx.flash(muzzle, Color(1, 0.8, 0.4), 4.0, 2)
		tracer_new = false


# ------------------------------------------------------------------ main loop (one-frame pipeline)
func _process(_d: float) -> void:
	if frames.is_empty():
		get_tree().quit()
		return
	if warm < 10:
		if warm == 0:
			_apply(0)
			_update_chars(0)
			_update_cams(true)
		RenderingServer.force_draw(false, dt)
		warm += 1
		return
	if pending:
		RenderingServer.force_draw(false, dt)
		fx.step()
		for entry in cams:
			var img: Image = (entry[1] as SubViewport).get_texture().get_image()
			img.save_jpg(out_dir.path_join("%s_%05d.jpg" % [entry[0], frame_i - 1]), 0.92)
		pending = false
	var last := mini(frames.size(), max_frames)
	if frame_i >= last:
		print("CINEMA_DONE frames=", frame_i)
		get_tree().quit()
		return
	_apply(frame_i)
	_update_chars(frame_i)
	_update_cams(false)
	frame_i += 1
	pending = true
