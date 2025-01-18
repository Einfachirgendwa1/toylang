mod compiler;
mod elf;
mod lexer;
mod linker;
mod misc;
mod parser;

use compiler::*;
use lexer::*;
use linker::*;
use misc::*;
use parser::*;

use std::{
    fs::{metadata, set_permissions, File},
    io::{Read, Write},
    os::unix::fs::PermissionsExt,
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use eyre::{Context, Result};
use log::{info, set_logger, set_max_level, warn, Level, LevelFilter, Log, SetLoggerError};

#[derive(Parser)]
struct Cli {
    #[arg(default_value = "main.tl")]
    input: String,

    #[arg(short, long, global = true)]
    output: Option<String>,
}

fn reverse(x: String) -> String {
    x.chars().rev().collect()
}

struct Logger {}

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

fn main() -> Result<()> {
    let Cli { input, output } = Cli::parse();

    use_logger(log::LevelFilter::Info).expect("Failed to set logger.");

    if !input.ends_with(".tl") {
        warn!("Input file `{input}` doesn't end in `.tl`. It should do that.")
    }

    let output = output.unwrap_or_else(|| {
        let mut chars = input.chars().rev().skip_while(|c| *c != '.');
        let next = chars.next();
        if next.is_none() {
            let output = format!("{input}.out");
            warn!("Input file `input` doesn't have a file extension.");
            warn!("Renaming output to {output} to not override input file.");
            output
        } else {
            reverse(chars.collect())
        }
    });

    info!("{input} --- tlc ---> {output}");

    let start = Instant::now();

    let mut input_file =
        File::open(&input).wrap_err_with(|| format!("Failed to open `{input}`."))?;

    let mut content = String::new();
    input_file
        .read_to_string(&mut content)
        .wrap_err_with(|| format!("Failed to read from `{input}`."))?;

    let tokens = tokenize(&content).wrap_err("Failed to tokenize.")?;
    let ast = parse(tokens).wrap_err("Failed to parse.")?;
    let elf = compile_main(ast).wrap_err("Failed to compile.")?;

    File::create(&output).unwrap().write_all(&elf).unwrap();
    let mut permissions = metadata(&output).unwrap().permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    set_permissions(output, permissions).unwrap();

    let diff = Instant::now().duration_since(start);
    info!("Finished successfully within {diff:?}.");

    Ok(())
}
