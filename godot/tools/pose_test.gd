# Close-up render of a Kenney character holding the blaster (for tuning cinema attachments).
extends SceneTree

func _init() -> void:
	var args := {}
	for a in OS.get_cmdline_user_args():
		var kv := a.substr(2).split("=", true, 1)
		args[kv[0]] = kv[1] if kv.size() > 1 else "1"
	var root := Node3D.new()
	get_root().add_child(root)
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.25, 0.28, 0.33)
	env.ambient_light_color = Color(0.7, 0.7, 0.75)
	env.ambient_light_energy = 0.8
	var we := WorldEnvironment.new()
	we.environment = env
	root.add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-40, -30, 0)
	root.add_child(sun)
	var inst: Node3D = (load("res://assets/kenney_mini/character-soldier.glb") as PackedScene).instantiate()
	inst.scale = Vector3.ONE * 2.35
	inst.rotation.y = PI
	root.add_child(inst)
	var ap: AnimationPlayer = inst.find_child("AnimationPlayer", true, false)
	ap.callback_mode_process = AnimationMixer.ANIMATION_CALLBACK_MODE_PROCESS_MANUAL
	var anim_name := str(args.get("anim", "holding-right"))
	ap.play(anim_name)
	ap.advance(float(args.get("t", 0.0)))
	var sk: Skeleton3D = inst.find_child("Skeleton3D", true, false)
	var ba := BoneAttachment3D.new()
	ba.bone_name = str(args.get("bone", "arm-right"))
	sk.add_child(ba)
	var g: Node3D = (load("res://assets/kenney_mini/weapon-shooter.glb") as PackedScene).instantiate()
	var p := str(args.get("pos", "0,0,0")).split(",")
	var r := str(args.get("rot", "0,0,0")).split(",")
	g.position = Vector3(float(p[0]), float(p[1]), float(p[2]))
	g.rotation_degrees = Vector3(float(r[0]), float(r[1]), float(r[2]))
	ba.add_child(g)
	# print arm track rotation for reference
	var an := ap.get_animation(anim_name)
	for i in an.get_track_count():
		if str(an.track_get_path(i)).ends_with(":arm-right"):
			print("TRACK ", an.track_get_path(i), " type=", an.track_get_type(i))
	var vp := SubViewport.new()
	vp.size = Vector2i(900, 500)
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.msaa_3d = Viewport.MSAA_4X
	root.add_child(vp)
	var views := [Vector3(1.1, 1.5, -1.9), Vector3(2.0, 1.3, 0.2)]
	var imgs := []
	for v in views:
		var cam := Camera3D.new()
		vp.add_child(cam)
		cam.current = true
		cam.look_at_from_position(v, Vector3(0, 1.1, 0))
		for k in 4:
			await process_frame
		imgs.append(vp.get_texture().get_image())
		cam.queue_free()
		await process_frame
	var out := Image.create(1800, 500, false, imgs[0].get_format())
	out.blit_rect(imgs[0], Rect2i(0, 0, 900, 500), Vector2i(0, 0))
	out.blit_rect(imgs[1], Rect2i(0, 0, 900, 500), Vector2i(900, 0))
	out.save_png(str(args.get("out", "/tmp/pose.png")))
	print("POSE_DONE")
	quit()
