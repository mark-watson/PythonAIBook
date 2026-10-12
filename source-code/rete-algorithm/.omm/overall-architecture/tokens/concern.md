`binding_dict` rebuilds a dict on every access and `wmes()` walks the full chain; both are called in hot paths (every join attempt, every retraction scan), so cost grows with rule length.
