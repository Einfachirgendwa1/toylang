use eyre::{ContextCompat, Result, WrapErr};

#[macro_export]
macro_rules! err {
    ($($expr:expr), *) => {
        return Err(eyre!($($expr), *))
    };
}

pub trait Impossible<T> {
    fn impossible(self) -> Result<T>;

    fn impossible_message(self, msg: &str) -> Result<T>;
}

impl<T, E> Impossible<T> for Result<T, E>
where
    Self: WrapErr<T, E>,
{
    #[cold]
    fn impossible(self) -> Result<T> {
        self.wrap_err("This error should be impossible to reach. This is a bug in the compiler.")
    }

    #[cold]
    fn impossible_message(self, msg: &str) -> Result<T> {
        self.wrap_err("This error should be impossible to reach. This is a bug in the compiler.")
            .wrap_err(msg.to_string())
    }
}

impl<T> Impossible<T> for Option<T> {
    #[cold]
    fn impossible(self) -> Result<T> {
        self.wrap_err("This error should be impossible to reach. This is a bug in the compiler.")
    }

    #[cold]
    fn impossible_message(self, msg: &str) -> Result<T> {
        self.wrap_err("This error should be impossible to reach. This is a bug in the compiler.")
            .wrap_err(msg.to_string())
    }
}
