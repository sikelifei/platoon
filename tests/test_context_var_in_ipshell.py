import pytest
from IPython.terminal.embed import InteractiveShellEmbed

from platoon.envs.codeact import executor_context


def _create_shell():
    """Utility to create a quiet InteractiveShellEmbed instance for testing."""
    # Creating a new shell each time avoids reusing global singleton state that
    # could be modified by previous tests.
    shell = InteractiveShellEmbed()
    # Make sure the banner and exit messages do not pollute the test output
    shell.banner1 = ""
    shell.banner2 = ""
    shell.exit_msg = ""
    return shell


def test_executor_context_visible_in_shell():
    """Executor context set in Python should be readable inside an IPython shell."""

    test_value = {"value": "simple"}
    token = executor_context.set(test_value)
    try:
        shell = _create_shell()
        # Inside the shell, read the value from the context var
        shell.run_cell("from platoon.envs.codeact import executor_context\ncontext_value = executor_context.get()")
        assert shell.user_ns["context_value"] == test_value
    finally:
        # Restore previous context to avoid leaking state between tests
        executor_context.reset(token)


def test_executor_context_nested_shells():
    """Context changes should propagate through nested IPython shells."""

    outer_value = {"level": "outer"}
    token = executor_context.set(outer_value)
    try:
        # ---- First (outer) shell ----
        shell1 = _create_shell()
        shell1.run_cell("from platoon.envs.codeact import executor_context\nouter_read = executor_context.get()")
        assert shell1.user_ns["outer_read"] == outer_value

        # Update the context inside the first shell
        shell1.run_cell("from platoon.envs.codeact import executor_context\nexecutor_context.set({'level': 'shell1'})")
        shell1.run_cell("from platoon.envs.codeact import executor_context\nshell1_read = executor_context.get()")
        assert shell1.user_ns["shell1_read"] == {"level": "shell1"}

        # ---- Second (nested) shell ----
        shell2 = _create_shell()
        shell2.run_cell("from platoon.envs.codeact import executor_context\nnested_initial = executor_context.get()")
        # The nested shell should see the value set by the first shell
        assert shell2.user_ns["nested_initial"] == {"level": "shell1"}

        # Update the context again inside the nested shell
        shell2.run_cell("from platoon.envs.codeact import executor_context\nexecutor_context.set({'level': 'shell2'})")
        shell2.run_cell("from platoon.envs.codeact import executor_context\nnested_updated = executor_context.get()")
        assert shell2.user_ns["nested_updated"] == {"level": "shell2"}

        # After the nested shell finishes, the first shell should now see the latest value
        shell1.run_cell("from platoon.envs.codeact import executor_context\nafter_nested = executor_context.get()")
        assert shell1.user_ns["after_nested"] == {"level": "shell2"}
    finally:
        # Always restore the original context to avoid leaking state
        executor_context.reset(token)


# ----- Asynchronous tests -----


@pytest.mark.asyncio
async def test_async_executor_context_visible_in_shell():
    """executor_context must be readable inside a shell via run_cell_async."""

    test_value = {"value": "async"}
    token = executor_context.set(test_value)
    try:
        shell = _create_shell()
        # Run cell asynchronously that writes the context value into user_ns
        await shell.run_cell_async(
            "from platoon.envs.codeact import executor_context\nasync_value = executor_context.get()"
        )
        assert shell.user_ns["async_value"] == test_value
    finally:
        executor_context.reset(token)


@pytest.mark.asyncio
async def test_async_executor_context_recursive_shell():
    """An IPython shell that spawns another shell should propagate ContextVar changes."""

    outer_value = {"level": "outer_async"}
    token = executor_context.set(outer_value)
    try:
        shell1 = _create_shell()

        # Confirm outer shell sees the initial value
        await shell1.run_cell_async(
            "from platoon.envs.codeact import executor_context\nouter_initial = executor_context.get()"
        )
        assert shell1.user_ns["outer_initial"] == outer_value

        # Spawn a nested shell *inside* the outer shell and update the context
        await shell1.run_cell_async(
            "import asyncio\n"
            "from IPython.terminal.embed import InteractiveShellEmbed\n"
            "from platoon.envs.codeact import executor_context\n"
            "inner_shell = InteractiveShellEmbed()\n"
            "inner_shell.banner1 = inner_shell.banner2 = inner_shell.exit_msg = ''\n"
            # inner reads current value
            "await inner_shell.run_cell_async('from platoon.envs.codeact import executor_context\\ninner_read = executor_context.get()')\n"  # noqa: E501
            # inner updates contextvar
            "await asyncio.create_task(inner_shell.run_cell_async('from platoon.envs.codeact import executor_context\\nexecutor_context.set({\\'level\\': \\'inner_async\\'})\\ncontext_value=executor_context.get()'))\n"  # noqa: E501
            # inner verifies and outer captures after update
            "outer_during_nested1 = inner_shell.user_ns['inner_read']\n"
            "outer_during_nested2 = inner_shell.user_ns['context_value']\n"
            "outer_after_nested = executor_context.get()\n"
        )

        assert shell1.user_ns["outer_during_nested1"] == {"level": "outer_async"}
        assert shell1.user_ns["outer_during_nested2"] == {"level": "inner_async"}
        assert shell1.user_ns["outer_after_nested"] == {"level": "outer_async"}

    finally:
        executor_context.reset(token)
