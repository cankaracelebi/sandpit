# Records arena 0 every `every` physics ticks to a compact binary replay for offline cinematic
# rendering (cinema mode). Format: one JSON header line, then float32 frames:
#   [tick] + per tracked node [pos(3) quat(4) scale(3) visible(1)] + per agent [vel(3) on_floor pitch hp flags 0]
# Tracked nodes = the arena's direct Node3D children (identical order when the arena is rebuilt
# from the same scenario script + params, which is what cinema mode does).
extends Node

const NODE_F := 11
const AGENT_F := 8

var arena
var file: FileAccess
var every := 2
var nodes: Array = []
var agent_nodes: Array = []
var flags_acc: Array = []


static func tracked_nodes(ar) -> Array:
	var out := []
	for c in ar.get_children():
		if c is Node3D:
			out.append(c)
	return out


func open(p_arena, path: String, p_every: int, meta: Dictionary) -> void:
	arena = p_arena
	every = max(1, p_every)
	nodes = tracked_nodes(arena)
	agent_nodes = arena.agents
	flags_acc.resize(agent_nodes.size())
	flags_acc.fill(0)
	var names := []
	for n in nodes:
		names.append(str(n.name))
	var agents := []
	for a in agent_nodes:
		agents.append({"node": nodes.find(a), "team": a.team, "policy": a.policy_name})
	var header := {
		"version": 1, "every": every, "node_floats": NODE_F, "agent_floats": AGENT_F,
		"n_nodes": nodes.size(), "n_agents": agent_nodes.size(), "nodes": names, "agents": agents,
	}
	header.merge(meta)
	file = FileAccess.open(path, FileAccess.WRITE)
	file.store_line(JSON.stringify(header))


func on_tick(tick: int) -> void:
	if file == null:
		return
	for i in agent_nodes.size():
		var a = agent_nodes[i]
		var f := 0
		if a.primary_pressed:
			f |= 4
		flags_acc[i] |= f
	if tick % every != 0:
		return
	var buf := PackedFloat32Array()
	buf.append(float(tick))
	for n in nodes:
		var t: Transform3D = n.transform
		var s := t.basis.get_scale()
		var q := t.basis.orthonormalized().get_rotation_quaternion()
		buf.append_array(PackedFloat32Array([t.origin.x, t.origin.y, t.origin.z, q.x, q.y, q.z, q.w, s.x, s.y, s.z,
			1.0 if n.visible else 0.0]))
	for i in agent_nodes.size():
		var a = agent_nodes[i]
		var flags: int = flags_acc[i] | (1 if a.alive else 0) | (2 if a.frozen else 0)
		flags |= int(arena.replay_flags(a)) << 4
		flags_acc[i] = 0
		buf.append_array(PackedFloat32Array([a.velocity.x, a.velocity.y, a.velocity.z, 1.0 if a.is_on_floor() else 0.0,
			a.pitch, a.hp, float(flags), 0.0]))
	file.store_buffer(buf.to_byte_array())


func close() -> void:
	if file != null:
		file.close()
		file = null
