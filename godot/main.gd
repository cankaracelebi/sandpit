# Entry scene: builds lighting + N arena copies of the chosen scenario, then hands control to
# RLSync. Also implements the human-play, screenshot and watch (spectator/recording) front-ends.
extends Node3D

const SCENARIOS := {
	"spike": "res://scenarios/spike/spike.gd",
	"shooter": "res://scenarios/shooter/shooter.gd",
	"sumo": "res://scenarios/sumo/sumo.gd",
	"tag": "res://scenarios/hideseek/hideseek.gd",
	"hideseek": "res://scenarios/hideseek/hideseek.gd",
	"parkour": "res://scenarios/parkour/parkour.gd",
}

var args := {}
var mode := "human"
var arenas: Array = []
var sun: DirectionalLight3D
var fp_cam: Camera3D
var hud: Label
var mouse_dx := 0.0
var mouse_dy := 0.0
var live_cam: Camera3D          # root-window camera in watch mode
var rec_overview: Array = []    # [SubViewport, Camera3D]
var rec_chase: Array = []
var record_dir := ""
var record_every := 2
var recorder: Node = null


func on_tick_end(tick: int) -> void:
	if recorder != null:
		recorder.on_tick(tick)


func on_quit() -> void:
	if recorder != null:
		recorder.close()


func _ready() -> void:
	args = RLSync.args
	mode = RLSync.mode
	var scen: String = RLSync.scenario
	var n_arenas := int(args.get("arenas", 1))
	var seed := int(args.get("seed", 0))
	var pov := int(args.get("pov", 64))
	var shadows := int(args.get("shadows", 0)) == 1
	var params := {}
	if args.has("params"):
		var parsed = JSON.parse_string(str(args["params"]).uri_decode())
		if parsed is Dictionary:
			params = parsed
	if scen == "tag" and not params.has("mode"):
		params["mode"] = "tag"
	if mode == "cinema":
		var cin = load("res://cinema/cinema.gd").new()
		add_child(cin)
		cin.start(self, args)
		return
	_build_environment(shadows)
	var script: Script = load(SCENARIOS[scen])
	for i in n_arenas:
		var ar = script.new()
		add_child(ar)
		ar.position = Vector3(i * ar.SPACING, 0, 0)
		ar.setup(i, seed, params, Vector2i(pov, pov), shadows)
		arenas.append(ar)
	RLSync.register(arenas, self)
	match mode:
		"human":
			_build_human_view()
		"watch":
			_build_watch_view()
			if args.has("replay_out"):
				recorder = load("res://cinema/replay_recorder.gd").new()
				add_child(recorder)
				recorder.open(arenas[0], str(args["replay_out"]), int(args.get("replay_every", 2)),
					{"scenario": scen, "params": params, "seed": seed, "action_repeat": RLSync.action_repeat})


func _build_environment(shadows: bool) -> void:
	var env := Environment.new()
	var sky := Sky.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.32, 0.5, 0.78)
	sky_mat.sky_horizon_color = Color(0.72, 0.8, 0.9)
	sky_mat.ground_bottom_color = Color(0.2, 0.2, 0.22)
	sky_mat.ground_horizon_color = Color(0.6, 0.62, 0.66)
	sky.sky_material = sky_mat
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 0.9
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, -35, 0)
	sun.light_energy = 1.1
	sun.shadow_enabled = shadows
	sun.directional_shadow_mode = DirectionalLight3D.SHADOW_ORTHOGONAL
	sun.directional_shadow_max_distance = 40.0
	add_child(sun)


# ------------------------------------------------------------------ human play
func _build_human_view() -> void:
	fp_cam = Camera3D.new()
	fp_cam.fov = 90
	fp_cam.near = 0.05
	fp_cam.cull_mask = RLSync.human_agent.cam.cull_mask
	add_child(fp_cam)
	fp_cam.current = true
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	_build_pov_inset()
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Label.new()
	hud.position = Vector2(10, 10)
	hud.add_theme_font_size_override("font_size", 14)
	layer.add_child(hud)
	var cross := Label.new()
	cross.text = "+"
	cross.add_theme_font_size_override("font_size", 22)
	cross.set_anchors_preset(Control.PRESET_CENTER)
	layer.add_child(cross)


func _build_pov_inset() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var n: int = min(arenas[0].agents.size(), 4)
	var win := DisplayServer.window_get_size()
	for i in n:
		var tr := TextureRect.new()
		tr.texture = arenas[0].agents[i].cam_vp.get_texture()
		tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		tr.stretch_mode = TextureRect.STRETCH_SCALE
		tr.size = Vector2(128, 128)
		tr.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		tr.position = Vector2(win.x - 138, 10 + i * 138)
		layer.add_child(tr)


func _unhandled_input(event: InputEvent) -> void:
	if mode != "human":
		return
	if event is InputEventMouseMotion:
		mouse_dx += event.relative.x
		mouse_dy += event.relative.y
	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		get_tree().quit()


func human_action() -> PackedInt32Array:
	var ag = RLSync.human_agent
	var a := PackedInt32Array([0, 0, 0, 0, 0, 0])
	if Input.is_key_pressed(KEY_W): a[0] = 1
	elif Input.is_key_pressed(KEY_S): a[0] = 2
	if Input.is_key_pressed(KEY_A): a[1] = 1
	elif Input.is_key_pressed(KEY_D): a[1] = 2
	if Input.is_key_pressed(KEY_SPACE): a[4] = 1
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT) or Input.is_key_pressed(KEY_E): a[5] = 1
	# mouse look is applied directly (finer than the discrete agent yaw bins)
	ag.yaw -= mouse_dx * 0.004
	ag.pitch = clampf(ag.pitch - mouse_dy * 0.004, -ag.pitch_limit, ag.pitch_limit)
	mouse_dx = 0.0
	mouse_dy = 0.0
	return a


func _process(_delta: float) -> void:
	if mode == "human" and RLSync.human_agent != null:
		var ag = RLSync.human_agent
		fp_cam.global_transform = ag.cam.global_transform
		hud.text = "%s | hp %.2f | return %.2f | t %d/%d\nWASD move, mouse look, SPACE jump, LMB/E primary, ESC quit" % [
			RLSync.scenario, ag.hp, ag.ep_return, ag.arena.t, ag.arena.max_ticks]
	elif mode == "watch" and live_cam != null:
		_pose_overview(live_cam, arenas[0])


# ------------------------------------------------------------------ cameras
func _push_cam(cam: Camera3D) -> void:
	# make a just-moved camera take effect before the next force_draw (no notification flush)
	RenderingServer.camera_set_transform(cam.get_camera_rid(), cam.global_transform)


func _pose_overview(cam: Camera3D, ar) -> void:
	var pose: Array = ar.spectator_pose()
	cam.look_at_from_position(ar.to_global(pose[0]), ar.to_global(pose[1]))
	_push_cam(cam)


func _pose_chase(cam: Camera3D, ag) -> void:
	var back: Vector3 = ag.global_transform.basis.z.normalized()
	var eye: Vector3 = ag.global_position + back * 3.2 + Vector3(0, 2.4, 0)
	var target: Vector3 = ag.global_position + Vector3(0, 1.2, 0) - back * 4.0
	cam.look_at_from_position(eye, target)
	_push_cam(cam)


func _make_capture_view(size: Vector2i, fov := 55.0) -> Array:
	var vp := SubViewport.new()
	vp.size = size
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.msaa_3d = Viewport.MSAA_4X
	add_child(vp)
	var cam := Camera3D.new()
	cam.fov = fov
	cam.far = 300
	vp.add_child(cam)
	cam.current = true
	return [vp, cam]


# ------------------------------------------------------------------ screenshots
func capture_screenshots() -> void:
	var out_dir := str(args.get("shot_dir", "user://shots"))
	var prefix := str(args.get("shot_name", RLSync.scenario))
	DirAccess.make_dir_recursive_absolute(out_dir)
	var ar = arenas[0]
	var spec := _make_capture_view(Vector2i(960, 540))
	_pose_overview(spec[1], ar)
	var chase := _make_capture_view(Vector2i(960, 540), 70.0)
	_pose_chase(chase[1], ar.agents[0])
	var top := _make_capture_view(Vector2i(540, 540))
	top[1].projection = Camera3D.PROJECTION_ORTHOGONAL
	top[1].size = ar.topdown_extent()
	var tc: Vector3 = ar.to_global(ar.topdown_center())
	top[1].look_at_from_position(tc + Vector3(0, 80, 0.001), tc)
	_push_cam(top[1])
	for ag in ar.agents:
		ag.sync_camera()
	await get_tree().process_frame  # let transform notifications reach the RenderingServer
	for i in 3:
		RenderingServer.force_draw(false, 1.0 / 60.0)
	spec[0].get_texture().get_image().save_png(out_dir.path_join(prefix + "_spectator.png"))
	chase[0].get_texture().get_image().save_png(out_dir.path_join(prefix + "_chase.png"))
	top[0].get_texture().get_image().save_png(out_dir.path_join(prefix + "_topdown.png"))
	var k := 0
	for ag in ar.agents:
		ag.cam_vp.get_texture().get_image().save_png(out_dir.path_join("%s_pov%d.png" % [prefix, k]))
		k += 1
	print("SCREENSHOTS_DONE ", out_dir)
	get_tree().quit()


# ------------------------------------------------------------------ watch / recording
func _build_watch_view() -> void:
	live_cam = Camera3D.new()
	live_cam.fov = 55
	live_cam.far = 300
	add_child(live_cam)
	live_cam.current = true
	_build_pov_inset()
	record_dir = str(args.get("record_dir", ""))
	record_every = int(args.get("record_every", 2))
	if record_dir != "":
		DirAccess.make_dir_recursive_absolute(record_dir)
		rec_overview = _make_capture_view(Vector2i(640, 360))
		rec_chase = _make_capture_view(Vector2i(640, 360), 70.0)


var _last_record_tick := -1000


func on_frame(tick: int) -> void:
	# called from RLSync._process (transforms already flushed) in watch mode
	if record_dir == "" or tick - _last_record_tick < record_every:
		return
	_last_record_tick = tick
	_pose_overview(rec_overview[1], arenas[0])
	_pose_chase(rec_chase[1], arenas[0].agents[0])
	# shadows only for the pretty recording cameras; agent POVs keep the training look
	var had := sun.shadow_enabled
	sun.shadow_enabled = int(args.get("record_shadows", 1)) == 1
	RenderingServer.force_draw(false, 1.0 / 60.0)
	sun.shadow_enabled = had
	rec_overview[0].get_texture().get_image().save_png(record_dir.path_join("%06d_overview.png" % tick))
	rec_chase[0].get_texture().get_image().save_png(record_dir.path_join("%06d_chase.png" % tick))
