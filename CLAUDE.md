# Claude Code Guidelines for Warp

## Build Commands

- **Full build**: `python build_lib.py` - Rebuilds all native libraries (warp.dll/libwarp.so and warp-clang.dll)
- **Quick build**: `python build_lib.py --quick` - Only rebuilds warp-clang, much faster for iteration. Use this when you haven't modified CUDA/C++ code in warp.cu or other .cu files.

Always prefer `--quick` during development iteration when possible.

## API Design Guidelines

- **Don't deprecate APIs during iteration**: When refactoring or simplifying code, don't add backward compatibility shims, legacy wrappers, or deprecation notices unless explicitly requested. Just replace the old API with the new one directly.
- **Keep it simple**: Avoid over-engineering. One clean API is better than multiple redundant ones for compatibility.

## C++ Code Style

- **K&R brace style**: The project uses clang-format which enforces K&R style (opening brace at end of line).
  ```cpp
  if (condition) {
      doSomething();
  }
  ```
- Pre-commit hooks run clang-format automatically, so don't fight the formatter.
