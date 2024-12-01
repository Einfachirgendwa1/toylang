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
};

use clap::Parser;
use common::use_logger;
use eyre::{Context, Result};
use log::{debug, info, warn};

#[derive(Parser)]
struct Cli {
    input: String,

    #[arg(short, long, global = true)]
    output: Option<String>,
}

fn reverse(x: String) -> String {
    x.chars().rev().collect()
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

    let mut input_file =
        File::open(&input).wrap_err_with(|| format!("Failed to open `{input}`."))?;

    let mut content = String::new();
    input_file
        .read_to_string(&mut content)
        .wrap_err_with(|| format!("Failed to read from `{input}`."))?;

    let tokens = tokenize(&content).wrap_err("Failed to tokenize.")?;
    debug!("Tokenizer Output: {tokens:?}");

    let ast = parse(tokens).wrap_err("Failed to parse.")?;
    debug!("Parser Output: {ast:?}");

    let elf = compile_main(ast).wrap_err("Failed to compile.")?;

    File::create(&output).unwrap().write_all(&elf).unwrap();
    let mut permissions = metadata(&output).unwrap().permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    set_permissions(output, permissions).unwrap();
    Ok(())
}
