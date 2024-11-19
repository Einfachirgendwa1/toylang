use std::{fs::File, io::Read, str::Chars};

use clap::Parser;
use common::use_recommended_logger;
use eyre::{Context, Result};
use log::{info, warn};

#[derive(Parser)]
struct Cli {
    input: String,

    #[arg(short, long, global = true)]
    output: Option<String>,
}

#[derive(Debug)]
enum Token {
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
    Int(i32),
    Ident(String),
}

fn reverse(x: String) -> String {
    x.chars().rev().collect()
}

fn main() -> Result<()> {
    let Cli { input, output } = Cli::parse();

    use_recommended_logger(log::LevelFilter::Info).expect("Failed to set logger.");

    if !input.ends_with(".tl") {
        warn!("Input file `{input}` doesn't end in `.tl`. It should do that.")
    }

    let output = output.unwrap_or_else(|| {
        let mut chars = input.chars().rev().skip_while(|c| *c != '.');
        let next = chars.next();
        if !next.is_some() {
            let output = format!("{input}.out");
            warn!("Input file `input` doesn't have a file extension.");
            warn!("Renaming output to {output} to not override input file.");
            return output;
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

    let tokens = dbg!(tokenize(&content));

    Ok(())
}

struct SkipWhitespace<'a>(Chars<'a>);

impl<'a> Iterator for SkipWhitespace<'a> {
    type Item = <Chars<'a> as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next()? {
            ' ' => self.next(),
            x => Some(x),
        }
    }
}

fn is_digit(c: &char) -> bool {
    c.is_digit(10)
}

fn is_normal_character(c: &char) -> bool {
    c.is_digit(36)
}

fn build_from_and_while(
    c: &char,
    i: &mut impl Iterator<Item = char>,
    f: impl Fn(&char) -> bool,
) -> String {
    let mut out = c.to_string();
    out.extend(i.by_ref().take_while(f));
    out
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut input = SkipWhitespace(input.chars());

    while let Some(next) = input.next() {
        let token = match next {
            '+' => Token::Add,
            '-' => Token::Sub,
            '*' => Token::Mul,
            '/' => Token::Div,
            '(' => Token::LParen,
            ')' => Token::RParen,
            ref x if is_digit(x) => Token::Int(
                build_from_and_while(x, &mut input, is_digit)
                    .parse()
                    .unwrap(),
            ),
            ref x if is_normal_character(x) => {
                Token::Ident(build_from_and_while(x, &mut input, is_normal_character))
            }
            ref x => Err(eyre::eyre!("Found an invalid character: {x}"))?,
        };

        tokens.push(token);
    }

    Ok(tokens)
}
