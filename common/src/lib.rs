use colored::Colorize;
use log::{set_logger, set_max_level, Level, LevelFilter, Log, SetLoggerError};

pub struct Logger {}

impl Log for Logger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!(
                "{}",
                match record.level() {
                    Level::Error => format!("[ERROR] {}", record.args()).red(),
                    Level::Warn => format!("[WARN ] {}", record.args()).yellow(),
                    Level::Info => format!("[INFO ] {}", record.args()).cyan(),
                    Level::Debug => format!("[DEBUG] {}", record.args()).green(),
                    Level::Trace => format!("[TRACE] {}", record.args()).black(),
                }
            );
        }
    }

    fn flush(&self) {}
}

pub fn use_logger(_level_filter: LevelFilter) -> Result<(), SetLoggerError> {
    set_logger(&Logger {})?;

    #[cfg(debug_assertions)]
    set_max_level(LevelFilter::Debug);

    #[cfg(not(debug_assertions))]
    set_max_level(_level_filter);

    Ok(())
}
