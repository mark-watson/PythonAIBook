An RHS called `ctx.halt()`. The current firing's mutations are still applied, then `run()` exits. Remaining instantiations stay in the conflict set, so a later `run()` continues from there.
