"""
sys.monitoring integration for TorchDynamo.

This module handles the setup and teardown of sys.monitoring callbacks for
Dynamo's code replacement mechanism. The low-level PY_START callback itself
is implemented in C (eval_frame.c) for performance, but the registration,
event setup, and cleanup logic is handled here in Python for clarity and
maintainability.

Instrumentation strategy: LOCAL events.

Rather than arming PY_START globally (``set_events``), which bumps CPython's
global instrumentation version and re-instruments ALL code on every change,
Dynamo instruments only the specific code objects it needs to intercept via
per-code ``set_local_events``. The callback + tool id are registered once and
kept for the process lifetime; instrumentation is then keyed on individual code
objects:

  * The entry code object (``fn.__code__`` of the ``torch.compile``-wrapped
    function) is instrumented from the ``eval_frame.py`` wrapper right before
    the function runs.
  * Resume-function code objects (from graph breaks) are instrumented when
    Dynamo generates them (``resume_execution.ContinueExecutionCache``).

Because we never arm global events, Dynamo's ``set_eval_frame`` lifecycle no
longer churns global instrumentation on every working-thread 0-crossing, and
teardown carries no process-wide tax on non-Dynamo code.
"""

import sys
import types
from typing import Optional


# Only available on Python 3.12+. sys is a built-in module, not a package, so
# `import sys.monitoring` fails; access it as an attribute instead.
if sys.version_info >= (3, 12):
    monitoring = sys.monitoring
else:
    monitoring = None  # type: ignore[assignment]


# Tool ID reserved for Dynamo. Using slot 3 (slots 3-5 are available for
# third-party tools; 6 is OPTIMIZER per PEP 669).
DYNAMO_SYS_MONITORING_TOOL_ID = 3

# Whether the tool id + callback have been registered with sys.monitoring.
_callback_registered = False

# Code objects we have already armed with local PY_START events, keyed by
# id(code). set_local_events re-instruments the code object each call, so we
# skip codes already armed to avoid needless churn. We must key on identity, not
# the code object itself: CPython code objects hash/compare by CONTENT, so two
# distinct-but-identical resume codes (e.g. regenerated after torch._dynamo.reset())
# would collide in a plain set, yet sys.monitoring arms events per code-object
# identity -- the second object would silently never get armed. The values keep a
# strong ref so an id() is never reused by a freed object.
_instrumented_codes: dict = {}


def is_sys_monitoring_available() -> bool:
    """Check if sys.monitoring is available (Python 3.12+)."""
    return monitoring is not None


def get_py_start_event() -> Optional[int]:
    """Get the PY_START event value from sys.monitoring.

    Returns None if sys.monitoring is not available.
    """
    if monitoring is None:
        return None
    try:
        return monitoring.events.PY_START
    except (AttributeError, ValueError):
        return None


def enable_sys_monitoring(callback_obj) -> bool:
    """Register Dynamo's sys.monitoring tool id and PY_START callback.

    Called from the C shim (``enable_sys_monitoring_shim``) on the first
    working-thread 0-crossing with the C callback object. This only performs
    one-time registration; it does NOT arm any events. Events are armed per
    code object via ``instrument_code``.

    Args:
        callback_obj: The C callback object created by the C code
                     (torch._C._dynamo.eval_frame module)

    Returns:
        True if the tool is registered (or already was), False otherwise.
    """
    global _callback_registered

    if monitoring is None:
        return False

    try:
        py_start_event = get_py_start_event()
        if py_start_event is None:
            return False

        if _callback_registered:
            return True

        try:
            monitoring.use_tool_id(DYNAMO_SYS_MONITORING_TOOL_ID, "torch.dynamo")
        except ValueError:
            # Tool ID already in use; that's OK if it's ours.
            pass

        monitoring.register_callback(
            DYNAMO_SYS_MONITORING_TOOL_ID,
            py_start_event,
            callback_obj,
        )

        _callback_registered = True
        return True
    except Exception:
        return False


def _arm_one(code, py_start_event) -> None:
    if id(code) in _instrumented_codes:
        return
    try:
        monitoring.set_local_events(
            DYNAMO_SYS_MONITORING_TOOL_ID, code, py_start_event
        )
        _instrumented_codes[id(code)] = code
    except Exception:
        # A code object may be un-instrumentable; instrumentation is best-effort.
        pass


def instrument_code(code, recursive: bool = True) -> None:
    """Arm PY_START local events for a code object (and nested code objects).

    Idempotent per code object. No-op unless the tool + callback have been
    registered (which the C shim does when Dynamo activates). This is the local
    counterpart to global ``set_events``: only the armed code objects fire
    PY_START, so the callback runs only for code Dynamo needs to intercept.

    With ``recursive=True`` (default), also arm every code object nested in
    ``code.co_consts``. Functions defined lexically inside the compiled function
    (closures, lambdas, comprehensions) have their code stored there; when they
    run as fresh frames after a graph break, Dynamo must intercept them the same
    way global events would. Over-arming a code object that ends up inlined is
    harmless: it simply never fires (inlined code has no frame), and if it does
    run as a skipped frame the C callback returns it unchanged.
    """
    if monitoring is None or not _callback_registered or code is None:
        return
    py_start_event = get_py_start_event()
    if py_start_event is None:
        return
    if not recursive:
        _arm_one(code, py_start_event)
        return

    stack = [code]
    while stack:
        cur = stack.pop()
        if id(cur) in _instrumented_codes:
            continue
        _arm_one(cur, py_start_event)
        for const in cur.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)


def disable_sys_monitoring() -> None:
    """Teardown hook for the C shim on working-thread 0-crossing.

    With local events there is nothing to tear down on a transient 0-crossing:
    only Dynamo-owned code objects are instrumented, so keeping them armed
    across the crossing costs nothing for non-Dynamo code (unlike global
    ``set_events``, which taxed the whole process). Leaving instrumentation in
    place also avoids re-instrumenting the same codes on the next call. The tool
    id and callback registration are likewise kept for the process lifetime.
    """
    return
