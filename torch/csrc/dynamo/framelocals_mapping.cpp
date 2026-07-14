#include <torch/csrc/dynamo/framelocals_mapping.h>

#include <c10/util/irange.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>

#define Py_BUILD_CORE
#include <internal/pycore_code.h>
#undef Py_BUILD_CORE

// Our own version of PyFrame_GetLocals.
// Also combines functionality from frame_init_get_vars and frame_get_var.
// PyFrame_GetLocals:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1213
// frame_init_get_vars:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1136
// frame_get_var:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1162
// PyFrame_GetLocals returns the frame locals dict.
// frame_init_get_vars initializes free variables from the closure.
// frame_get_var fetches the variable value from the frame given the index
// NOTE: hidden variables are not included.
// Returns a new reference.
FrameLocalsMapping::FrameLocalsMapping(FrameLocalsFrameType* frame)
    : _code_obj(py::cast<py::object>((PyObject*)F_CODE(frame))) {
  PyCodeObject* co = F_CODE(frame);
  auto n = co->co_nlocalsplus;
  _framelocals.resize(n);

#if IS_PYTHON_3_16_PLUS
  TORCH_CHECK(false, "Python 3.16+");
#elif IS_PYTHON_3_14_PLUS
  // Let CPython extract the localsplus values: cell/free unboxing, hidden-var
  // skipping, not-yet-started frames, and pre-COPY_FREE_VARS resolution of free
  // variables from the closure are all handled by the helper. Values come back
  // as strong references indexed by localsplus index.
  std::vector<PyObject*> values(n, nullptr);
  if (PyUnstable_InterpreterFrame_GetLocals(frame, values.data()) < 0) {
    throw py::error_already_set();
  }
  for (int i = 0; i < n; i++) {
    _framelocals[i] = py::reinterpret_steal<py::object>(values[i]);
  }
#else
  if (!frame->stacktop) {
    return;
  }

  auto update_framelocals = [&](int i, PyObject* value) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
      return;
    }

#if IS_PYTHON_3_12_PLUS
    if (kind & CO_FAST_HIDDEN) {
      return;
    }
#endif

    if (kind & CO_FAST_FREE) {
      CHECK(value != nullptr && PyCell_Check(value));
      value = PyCell_GET(value);
    }

    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    _framelocals[i] = py::reinterpret_borrow<py::object>(value);
  };

  auto offset = co->co_nlocalsplus - co->co_nfreevars;
  for (const auto i : c10::irange(offset)) {
    update_framelocals(i, frame->localsplus[i]);
  }

  // Get references to closure variables
  PyObject* closure = FUNC(frame)->func_closure;
  for (const auto i : c10::irange(co->co_nfreevars)) {
    update_framelocals(offset + i, PyTuple_GET_ITEM(closure, i));
  }

  // NOTE no need to move the instruction pointer to after COPY_FREE_VARS
  // since we don't actually copy free vars from the closure to the frame
  // localsplus.
#endif
}

void FrameLocalsMapping::_realize_dict() {
  _dict = py::dict();
  py::tuple framelocals_names = code_framelocals_names(_code_obj);

  auto nlocalsplus = ((PyCodeObject*)_code_obj.ptr())->co_nlocalsplus;
  DEBUG_CHECK(nlocalsplus == _framelocals.size());
  for (const auto i : c10::irange(nlocalsplus)) {
    if (_framelocals[i]) {
      _dict[framelocals_names[i]] = _framelocals[i];
    }
  }
}

py::tuple code_framelocals_names(py::handle code) {
  CHECK(PyCode_Check(code.ptr()));
  return py::cast<py::tuple>(((PyCodeObject*)code.ptr())->co_localsplusnames);
}

PyObject* FrameLocalsMapping::get(int idx) {
  DEBUG_CHECK(0 <= idx && idx < _framelocals.size());
  return _framelocals[idx].ptr();
}

PyDictObject* framelocals_mapping_to_dict(FrameLocalsMapping* map) {
  return map->to_dict();
}
