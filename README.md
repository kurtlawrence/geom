# Testing

Usually testing is done via `cargo test`.

Some tests use massive test files which are not committed to avoid blowing out the
repository size.
These files are housed locally and ignored in the tests.
The tests can be run with `cargo test -- --ignored`.