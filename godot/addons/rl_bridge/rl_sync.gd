# RLSync (autoload): drives the simulation and talks to Python in lockstep.
#
# Rendering is explicit: when a decision is due we call RenderingServer.force_draw() and read the
# agents' camera viewports immediately. The exchange happens in _process, i.e. after this
# main-loop iteration's physics ticks have run *and* Godot has flushed the transform
# notifications to the RenderingServer. (Rendering from inside _physics_process would show moving
# meshes/cameras one tick stale.) _physics_process stops simulating once `action_repeat` ticks have
# run since the last decision, so each decision covers exactly `action_repeat` ticks.
# Nothing depends on the OS window being visible. macOS stops drawing occluded windows, which
# would stall a design that waits for frame signals.
# In train mode the normal render loop is disabled. Physics runs at 60*speedup ticks per real
# second with time_scale=speedup, so each tick is still 1/60 s of game time.
#
# Wire format (both directions): [u32 body_len][u32 header_len][header JSON][binary payload]
extends Node

enum State { IDLE, HANDSHAKE, RUN }

var args := {}
var mode := "human"          # train | watch | human | screenshot
var scenario := "spike"
var action_repeat := 4
var speedup := 1000.0
var arenas: Array = []
var agents: Array = []
var state := State.IDLE
var peer: StreamPeerTCP
var ticks_since := 0
var total_ticks := 0
var decisions := 0
var main_node: Node = null
var rand := RandomNumberGenerator.new()
var human_agent = null
var _random_hold := 0
var _prof := {"render": 0, "read": 0, "net": 0, "sim": 0, "n": 0}


func _ready() -> void:
	process_physics_priority = -100
	args = parse_args()
	mode = str(args.get("mode", "human"))
	scenario = str(args.get("scenario", "spike"))
	action_repeat = int(args.get("action_repeat", 4))
	speedup = float(args.get("speedup", 1000.0))
	rand.seed = int(args.get("seed", 0)) + 12345


static func parse_args() -> Dictionary:
	var d := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv := a.substr(2).split("=", true, 1)
			d[kv[0]] = kv[1] if kv.size() > 1 else "1"
	return d


func _fast_mode() -> void:
	Engine.physics_ticks_per_second = int(60.0 * speedup)
	Engine.time_scale = speedup
	Engine.max_physics_steps_per_frame = action_repeat
	RenderingServer.render_loop_enabled = false
	OS.low_processor_usage_mode_sleep_usec = 0
	# NOTE: the root viewport must stay active, otherwise its child camera SubViewports are not
	# drawn. It renders no 3D (no root camera / disable_3d), so it costs almost nothing.
	get_tree().root.disable_3d = true
	if int(args.get("minimize", 0)) == 1:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MINIMIZED)


func register(p_arenas: Array, p_main: Node) -> void:
	arenas = p_arenas
	main_node = p_main
	agents.clear()
	for ar in arenas:
		for ag in ar.agents:
			ag.index = agents.size()
			agents.append(ag)
	match mode:
		"train", "screenshot":
			_fast_mode()
		"watch":
			if int(args.get("realtime", 1)) == 1:
				Engine.physics_ticks_per_second = 60
				Engine.time_scale = 1.0
			else:
				_fast_mode()
				# one main-loop iteration per recorded frame
				Engine.max_physics_steps_per_frame = mini(action_repeat, int(args.get("record_every", 2)))
		_:
			Engine.physics_ticks_per_second = 60
			Engine.time_scale = 1.0
	if mode == "train" or mode == "watch":
		if str(args.get("pov_read", "atlas")) == "atlas" and agents.size() > 0:
			_build_atlas()
		_connect(int(args.get("port", 11008)))
		_send({"type": "hello", "version": 2, "env_info": env_info()})
		state = State.HANDSHAKE
	else:
		state = State.RUN
		if agents.size() > 0:
			human_agent = agents[0]


# All agent camera viewports become children of one 2D "atlas" SubViewport that tiles their
# textures 1:1, so a single GPU readback per decision returns every agent's image (Python slices
# it). Child viewports are always drawn before their parent, so the atlas is never stale.
var atlas: SubViewport = null
var atlas_grid := Vector2i(1, 1)  # (cols, rows)


func _build_atlas() -> void:
	var n := agents.size()
	var sz: Vector2i = agents[0].cam_vp.size
	var cols := int(ceil(sqrt(float(n))))
	var rows := int(ceil(float(n) / cols))
	atlas_grid = Vector2i(cols, rows)
	atlas = SubViewport.new()
	atlas.size = Vector2i(cols * sz.x, rows * sz.y)
	atlas.disable_3d = true
	atlas.transparent_bg = false
	atlas.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	main_node.add_child(atlas)
	for i in n:
		var vp: SubViewport = agents[i].cam_vp
		vp.get_parent().remove_child(vp)
		atlas.add_child(vp)
		var tr := TextureRect.new()
		tr.texture = vp.get_texture()
		tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		tr.stretch_mode = TextureRect.STRETCH_SCALE
		tr.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		tr.position = Vector2((i % cols) * sz.x, (i / cols) * sz.y)
		tr.size = Vector2(sz)
		atlas.add_child(tr)


func env_info() -> Dictionary:
	var ag_list := []
	for ag in agents:
		ag_list.append({"arena": ag.arena.arena_id, "team": ag.team, "policy": ag.policy_name})
	var pov: Vector2i = arenas[0].pov_size if arenas.size() > 0 else Vector2i(64, 64)
	return {
		"scenario": scenario,
		"n_agents": agents.size(),
		"n_arenas": arenas.size(),
		"pov_shape": [pov.y, pov.x, 3],
		"vec_dim": arenas[0].vec_dim() if arenas.size() > 0 else 0,
		"nvec": [3, 3, 5, 3, 2, 2],
		"action_repeat": action_repeat,
		"agents": ag_list,
		"params": arenas[0].params if arenas.size() > 0 else {},
	}


# ---------------------------------------------------------------- simulation tick
func _physics_process(delta: float) -> void:
	if state != State.RUN:
		return
	var synchronous := mode == "train" or mode == "watch"
	if synchronous and ticks_since >= action_repeat:
		return  # decision pending (handled in _process)
	if not synchronous:
		_local_actions()
	var t0 := Time.get_ticks_usec()
	for ar in arenas:
		if ar.skip_rest:
			continue
		ar.t += 1
		ar.pre_tick(delta)
		for ag in ar.agents:
			ag.physics_step(delta)
		ar.post_tick(delta)
		if not ar.ended and ar.t >= ar.max_ticks:
			ar.end_episode(true)
		if ar.ended:
			for ag in ar.agents:
				ag.done = true
				ag.truncated = ar.ended_truncated
				ag.info = ar.episode_info(ag)
			ar.do_reset()
			if synchronous:
				ar.skip_rest = true
	ticks_since += 1
	total_ticks += 1
	_prof["sim"] += Time.get_ticks_usec() - t0
	if mode == "watch" and main_node.has_method("on_tick_end"):
		main_node.on_tick_end(total_ticks)
	if not synchronous:
		for ag in agents:
			ag.reward = 0.0
			ag.done = false
			ag.info = null
		if mode == "screenshot" and total_ticks == int(args.get("shot_ticks", 120)):
			state = State.IDLE
			main_node.capture_screenshots.call_deferred()


func _process(_delta: float) -> void:
	if state == State.HANDSHAKE:
		_recv_loop()
		return
	if state != State.RUN or not (mode == "train" or mode == "watch"):
		return
	if mode == "watch" and main_node.has_method("on_frame"):
		main_node.on_frame(total_ticks)
	if ticks_since >= action_repeat:
		_render()
		_send_obs()
		_recv_loop()


func _render() -> void:
	var t0 := Time.get_ticks_usec()
	RenderingServer.force_draw(false, 1.0 / 60.0)
	_prof["render"] += Time.get_ticks_usec() - t0


func _local_actions() -> void:
	# human / screenshot modes: agent 0 from keyboard (human), everyone else random.
	_random_hold -= 1
	for ag in agents:
		if ag == human_agent and mode == "human":
			ag.action = main_node.human_action()
		elif _random_hold <= 0:
			ag.action = random_action()
	if _random_hold <= 0:
		_random_hold = action_repeat * 3


func random_action() -> PackedInt32Array:
	var a := PackedInt32Array()
	for n in [3, 3, 5, 3, 2, 2]:
		a.append(rand.randi_range(0, n - 1))
	if rand.randf() < 0.5:
		a[0] = 1  # bias forward so random play explores
	if rand.randf() < 0.7:
		a[3] = 0
	return a


# ---------------------------------------------------------------- python exchange
func _send_obs() -> void:
	var t0 := Time.get_ticks_usec()
	var pov := PackedByteArray()
	if atlas != null:
		var img := atlas.get_texture().get_image()
		if img.get_format() != Image.FORMAT_RGB8:
			img.convert(Image.FORMAT_RGB8)
		pov = img.get_data()
	var blind := PackedByteArray()
	var vec := PackedFloat32Array()
	var rew := PackedFloat32Array()
	var dn := PackedByteArray()
	var tr := PackedByteArray()
	var infos := []
	for ag in agents:
		if atlas == null:
			pov.append_array(ag.pov_bytes())
		blind.append(1 if ag.blind else 0)
		vec.append_array(PackedFloat32Array(ag.arena.get_vec(ag)))
		rew.append(ag.reward)
		dn.append(1 if ag.done else 0)
		tr.append(1 if ag.truncated else 0)
		if ag.info != null:
			var d: Dictionary = ag.info
			d["agent"] = ag.index
			infos.append(d)
		ag.reward = 0.0
		ag.done = false
		ag.truncated = false
		ag.info = null
	_prof["read"] += Time.get_ticks_usec() - t0
	var payload := pov
	payload.append_array(vec.to_byte_array())
	payload.append_array(rew.to_byte_array())
	payload.append_array(dn)
	payload.append_array(tr)
	payload.append_array(blind)
	decisions += 1
	var hdr := {"type": "obs", "infos": infos, "tick": total_ticks, "decision": decisions,
		"pov_layout": "atlas" if atlas != null else "agents", "atlas_grid": [atlas_grid.y, atlas_grid.x]}
	_send(hdr, payload)


func _recv_loop() -> void:
	var t0 := Time.get_ticks_usec()
	var waiting := true
	while waiting:
		var msg := _recv()
		var h: Dictionary = msg[0]
		var payload: PackedByteArray = msg[1]
		match str(h.get("type", "")):
			"step":
				var acts := payload.to_int32_array()
				for i in agents.size():
					agents[i].action = acts.slice(i * 6, i * 6 + 6)
				for ar in arenas:
					ar.skip_rest = false
				ticks_since = 0
				state = State.RUN
				waiting = false
			"reset":
				if h.has("seed"):
					for ar in arenas:
						ar.rng.seed = hash([int(h["seed"]), ar.arena_id])
				for ar in arenas:
					ar.do_reset()
					ar.skip_rest = false
				for ag in agents:
					ag.reward = 0.0
					ag.done = false
					ag.truncated = false
					ag.info = null
				ticks_since = 0
				_render()
				_send_obs()
			"call":
				var result = _handle_call(str(h.get("method", "")), h.get("args", []))
				_send({"type": "ack", "result": result})
			"close":
				_quit()
				return
			_:
				push_error("RLSync: unknown message %s" % h)
				_quit()
				return
	_prof["net"] += Time.get_ticks_usec() - t0
	_prof["n"] += 1
	if args.has("profile") and _prof["n"] % 200 == 0:
		print("PROFILE per decision (ms): sim %.2f render %.2f read %.2f wait-python %.2f" % [
			_prof["sim"] / 200000.0, _prof["render"] / 200000.0, _prof["read"] / 200000.0, _prof["net"] / 200000.0])
		for k in ["sim", "render", "read", "net"]:
			_prof[k] = 0


func _handle_call(method: String, call_args: Array):
	match method:
		"set_params":
			var p: Dictionary = call_args[0]
			var ids = call_args[1] if call_args.size() > 1 else null
			for ar in arenas:
				if ids == null or ar.arena_id in ids:
					ar.set_params(p)
			return true
		"env_info":
			return env_info()
		"ping":
			return "pong"
	if main_node.has_method("handle_call"):
		return main_node.handle_call(method, call_args)
	return null


# ---------------------------------------------------------------- socket helpers
func _connect(port: int) -> void:
	for attempt in 200:
		peer = StreamPeerTCP.new()
		peer.connect_to_host("127.0.0.1", port)
		var t0 := Time.get_ticks_msec()
		while peer.get_status() == StreamPeerTCP.STATUS_CONNECTING and Time.get_ticks_msec() - t0 < 2000:
			peer.poll()
			OS.delay_msec(2)
		if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			peer.set_no_delay(true)
			return
		OS.delay_msec(50)
	push_error("RLSync: could not connect to python on port %d" % port)
	_quit()


func _send(header: Dictionary, payload := PackedByteArray()) -> void:
	var hb := JSON.stringify(header).to_utf8_buffer()
	var pre := PackedByteArray()
	pre.resize(8)
	pre.encode_u32(0, 4 + hb.size() + payload.size())
	pre.encode_u32(4, hb.size())
	pre.append_array(hb)
	pre.append_array(payload)
	if peer.put_data(pre) != OK:
		_quit()


func _recv() -> Array:
	var r: Array = peer.get_data(4)
	if r[0] != OK:
		_quit()
		return [{"type": "close"}, PackedByteArray()]
	var body_len: int = (r[1] as PackedByteArray).decode_u32(0)
	var r2: Array = peer.get_data(body_len)
	if r2[0] != OK:
		_quit()
		return [{"type": "close"}, PackedByteArray()]
	var body: PackedByteArray = r2[1]
	var hlen := body.decode_u32(0)
	var header = JSON.parse_string(body.slice(4, 4 + hlen).get_string_from_utf8())
	return [header, body.slice(4 + hlen)]


func _quit() -> void:
	state = State.IDLE
	if main_node != null and main_node.has_method("on_quit"):
		main_node.on_quit()
	if peer != null:
		peer.disconnect_from_host()
	get_tree().quit()
