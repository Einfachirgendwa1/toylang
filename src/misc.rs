use eyre::{ContextCompat, Result, WrapErr};

pub trait Impossible<T> {
    fn impossible(self) -> Result<T>;
}

impl<T> Impossible<T> for Result<T> {
    fn impossible(self) -> Result<T> {
        self.wrap_err("This error should be impossible to reach. This is a bug in the compiler.")
    }
}

impl<T> Impossible<T> for Option<T> {
    fn impossible(self) -> Result<T> {
        self.wrap_err("This error should be impossible to reach. This is a bug in the compiler.")
    }
}
