# NetBeans project wrapper

NetBeans has no first-party Rust project type, so this is set up as a
**Makefile-based C/C++ project** (`nbproject/project.xml` type
`org.netbeans.modules.cnd.makeproject`, `nbproject/configurations.xml`
conf `type="3"`). NetBeans indexes/browses the `.rs` files as generic
source text, and all build/run/clean actions delegate to `cargo` via the
top-level `Makefile`:

- **Build** -> `cargo build`
- **Clean** -> `cargo clean`
- **Run** -> `cargo run` (binary at `target/debug/expr-minimizer`)
- `make test` / `cargo test` for the unit tests (not wired to a NetBeans
  menu action by default — run from the Terminal panel, or add a custom
  action in Project Properties > Build Actions if you want a menu item).

## Opening it

File -> Open Project... -> select the `expr-minimizer` folder (the one
containing `nbproject/`).

Don't expect Rust-aware syntax highlighting, code completion, or inline
error checking from NetBeans itself — it's just editing/browsing +
delegated build here. For real Rust tooling (rust-analyzer-grade
completion, inline diagnostics), IntelliJ/CLion with the Rust plugin, VS
Code with rust-analyzer, or `rustc`'s own `cargo check`/`clippy` on the
command line will serve you much better than NetBeans' C/C++ tooling
pretending to understand `.rs` files.

## Caveat

I hand-authored `nbproject/project.xml` and `nbproject/configurations.xml`
to match the known NetBeans Makefile-project schema, but I don't have a
NetBeans instance available here to actually open and verify this against.
The XML is well-formed (checked), and the structure mirrors what NetBeans
itself generates for "C/C++ Project with Existing Sources -> Configure
Build -> Custom" — but if NetBeans complains or silently drops something on
first open, the fix is usually to let NetBeans regenerate
`nbproject/private/` itself (already gitignored) and to double-check the
`<conf name="...">` / `<item path="...">` entries in `configurations.xml`
against what your specific NetBeans version expects.
