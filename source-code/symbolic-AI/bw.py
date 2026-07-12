# bw.py - Soar Cognitive Architecture blocks world example
#
# Runs the classic blocks world problem using the Soar production rule engine.
# The agent starts with three blocks (A, B, C) on a table and moves them
# randomly until they are stacked in the goal configuration: A on B on C.
#
# The Soar production rules in bw.soar define:
#   - Initial state creation (blocks on table)
#   - Operator proposals (which block moves are legal)
#   - Operator applications (update working memory after a move)
#   - Goal detection (halt when A-B-C tower is achieved)
#
# Requirements: uv pip install soar-sml

import soar_sml as sml


def callback_debug(mid, user_data, agent, message):
    """Print callback: receives output from Soar's print events."""
    print(message)


if __name__ == "__main__":
    # Create the Soar kernel and a named agent
    soar_kernel = sml.Kernel.CreateKernelInCurrentThread()
    soar_agent = soar_kernel.CreateAgent("agent")

    # Register a callback so we see the agent's output (move announcements, etc.)
    soar_agent.RegisterForPrintEvent(sml.smlEVENT_PRINT, callback_debug, None)

    # Load the blocks world production rules
    soar_agent.ExecuteCommandLine("source bw.soar")

    # Run up to 50 decision cycles (the 3-block problem solves in far fewer)
    run_result = soar_agent.RunSelf(50)

    # Clean up
    soar_kernel.DestroyAgent(soar_agent)
    soar_kernel.Shutdown()
