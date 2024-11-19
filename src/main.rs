#![feature(yeet_expr)]

use std::{fmt::Display, fs::File, io::Read, str::Chars};

use clap::Parser;
use common::use_recommended_logger;
use eyre::{eyre, Context, Report, Result};
use log::{error, info, warn};

#[derive(Parser)]
struct Cli {
    input: String,

    #[arg(short, long, global = true)]
    output: Option<String>,
}

#[derive(Clone, Debug)]
enum Token {
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
    Equal,
    Int(i32),
    Ident(String),
    Let,
    Semi,
    EOF,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Add => write!(f, "+"),
            Token::Sub => write!(f, "-"),
            Token::Mul => write!(f, "*"),
            Token::Div => write!(f, "/"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Equal => write!(f, "="),
            Token::Semi => write!(f, ";"),
            Token::Let => write!(f, "the `let` keyword"),
            Token::Int(x) => write!(f, "an integer ({x})"),
            Token::Ident(x) => write!(f, "an identifier (\"{x}\")"),
            Token::EOF => write!(f, "the end of the file"),
        }
    }
}

struct AST {
    statements: Vec<Statement>,
}

enum Statement {
    FunctionDefinition {
        function_name: String,
        paramters: Vec<String>,
    },
    VariableAssignment {
        variable_name: String,
        value: Expression,
    },
    Return {
        value: Expression,
    },
    Expression {
        value: Expression,
    },
    CodeBlock {
        ast: AST,
    },
    NullOpt,
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::VariableAssignment {
                variable_name,
                value,
            } => write!(f, "assignment of {value} to `{variable_name}`"),
            Statement::Return { value } => write!(f, "return {value}"),
            Statement::Expression { value } => write!(f, "expression {value}"),
            Statement::FunctionDefinition {
                function_name,
                paramters: _,
            } => write!(f, "definition of function `{function_name}`"),
            Statement::CodeBlock { ast: _ } => write!(f, "code block"),
            Statement::NullOpt => write!(f, "empty statement"),
        }
    }
}

#[derive(Clone)]
enum Expression {
    FunctionCall {
        function_name: String,
        arguments: Vec<Expression>,
    },
    MathematicalOperation {
        left_side: Box<Expression>,
        right_side: Box<Expression>,
        operator: Operator,
    },
    LiteralInt {
        value: i32,
    },
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::FunctionCall {
                function_name,
                arguments: _,
            } => write!(f, "call of function `{function_name}`)"),
            Expression::MathematicalOperation {
                left_side,
                right_side,
                operator,
            } => write!(f, "{left_side} {operator} {right_side}"),
            Expression::LiteralInt { value } => write!(f, "{value}"),
        }
    }
}

#[derive(Clone)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add => write!(f, "+"),
            Operator::Sub => write!(f, "-"),
            Operator::Mul => write!(f, "*"),
            Operator::Div => write!(f, "/"),
        }
    }
}

impl TryFrom<Token> for Operator {
    type Error = Report;

    fn try_from(value: Token) -> std::result::Result<Self, Self::Error> {
        match value {
            Token::Add => Ok(Operator::Add),
            Token::Sub => Ok(Operator::Sub),
            Token::Mul => Ok(Operator::Mul),
            Token::Div => Ok(Operator::Div),
            x => do yeet eyre!("Cannot convert {x} into an operator."),
        }
    }
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

    let tokens = tokenize(&content).wrap_err("Failed to tokenize.")?;
    let ast = parse(tokens).wrap_err("Failed to parse.")?;

    Ok(())
}

struct SkipWhitespace<'a>(Chars<'a>);

impl<'a> Iterator for SkipWhitespace<'a> {
    type Item = <Chars<'a> as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next()? {
            ' ' | '\n' | '\t' => self.next(),
            x => Some(x),
        }
    }
}

impl<'a> SkipWhitespace<'a> {
    fn temporary_dont_skip(&mut self) -> &mut Chars<'a> {
        &mut self.0
    }
}

fn is_digit(c: &char) -> bool {
    c.is_digit(10)
}

fn is_normal_character(c: &char) -> bool {
    c.is_digit(36)
}

fn build_string_from_init_and_while(
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
            '=' => Token::Equal,
            ';' => Token::Semi,
            ref x if is_digit(x) => Token::Int(
                build_string_from_init_and_while(x, &mut input, is_digit)
                    .parse()
                    .unwrap(),
            ),
            ref x if is_normal_character(x) => {
                let string = build_string_from_init_and_while(
                    x,
                    input.temporary_dont_skip(),
                    is_normal_character,
                );

                match string.as_str() {
                    "let" => Token::Let,
                    _ => Token::Ident(string),
                }
            }
            ref x => Err(eyre!(
                "Found an invalid character: {x} (character code {})",
                *x as u8
            ))?,
        };

        tokens.push(token);
    }

    tokens.push(Token::EOF);

    Ok(tokens)
}

fn get_next_statement<'a>(tokens: &mut impl Iterator<Item = Token>) -> Result<Option<Statement>> {
    let Some(next) = tokens.next() else {
        return Ok(None);
    };

    let statement = match next {
        Token::Let => {
            let variable_name = match tokens.next().unwrap() {
                Token::Ident(ident) => ident,
                x => do yeet eyre!("Expected identifier after let, found {x}"),
            };

            match tokens.next().unwrap() {
                Token::Equal => (),
                x => {
                    do yeet eyre!("Exprected equal sign (=) after `let {variable_name}`, found {x}")
                }
            }

            let value = match get_next_statement(tokens) {
                Err(err) => Err(err).wrap_err_with(|| format!("Failed to get ... in `let {variable_name} = ...`") )?,
                Ok(None) => {
                    do yeet eyre!(
                    "Expected an expression to follow after `let {variable_name} = `, but got nothing."
                )
                }
                Ok(Some(Statement::Expression { value })) => value,
                Ok(Some(x)) => {
                    do yeet eyre!(
                        "Expected an expression to follow after `let {variable_name} = `, but got {x}."
                    )
                }
            };

            Statement::VariableAssignment {
                variable_name,
                value,
            }
        }

        // TODO: Remove duplication
        Token::Add => match get_next_statement(tokens) {
            Err(err) => {
                Err(err).wrap_err_with(|| format!("Failed to get the value after the `+`"))?
            }
            Ok(None) => do yeet eyre!("Unexpected `+` with nothing after it."),
            Ok(Some(Statement::Expression { value })) => Statement::Expression { value },
            Ok(Some(y)) => do yeet eyre!("Expected Expression after `+`, got {y}"),
        },
        Token::Sub => match get_next_statement(tokens) {
            Err(err) => {
                Err(err).wrap_err_with(|| format!("Failed to get the value after the `-`"))?
            }
            Ok(None) => do yeet eyre!("Unexpected `-` with nothing after it."),
            Ok(Some(Statement::Expression { value })) => Statement::Expression {
                value: Expression::MathematicalOperation {
                    left_side: Box::new(value),
                    right_side: Box::new(Expression::LiteralInt { value: -1 }),
                    operator: Operator::Mul,
                },
            },
            Ok(Some(y)) => do yeet eyre!("Expected Expression after `-`, got {y}"),
        },

        Token::LParen => {
            let mut tokens_in_parenthesis = Vec::new();
            loop {
                match tokens.next() {
                    None => do yeet eyre!("Parenthesis was never closed."),
                    Some(Token::LParen) => break,
                    Some(x) => tokens_in_parenthesis.push(x),
                }
            }

            Statement::CodeBlock {
                ast: parse(tokens_in_parenthesis)
                    .wrap_err("Failed to parse tokens in parenthesis.")?,
            }
        }

        x @ (Token::Mul | Token::Div | Token::Equal | Token::RParen) => {
            do yeet eyre!("Detected junk: `{x}`")
        }

        Token::EOF => return Ok(None),

        Token::Int(value) => match tokens.next() {
            None | Some(Token::Semi) => Statement::Expression {
                value: Expression::LiteralInt { value },
            },
            Some(x @ (Token::Add | Token::Sub | Token::Mul | Token::Div)) => {
                let expr = match get_next_statement(tokens).wrap_err_with(|| {
                    format!("Failed to get right hand side of `{value} {x} ...`")
                })? {
                    None => do yeet eyre!("Expected right hand side of expression, found nothing."),
                    Some(Statement::Expression { value }) => value,
                    Some(x) => {
                        do yeet eyre!(
                            "Exprected right side of expression to be an expression, found {x}"
                        )
                    }
                };

                Statement::Expression {
                    value: Expression::MathematicalOperation {
                        left_side: Box::new(Expression::LiteralInt { value }),
                        right_side: Box::new(expr.clone()),
                        operator: x.clone().try_into().wrap_err_with(|| {
                            format!("Failed to construct an expression out of `{value} {x} {expr}`")
                        })?,
                    },
                }
            }

            Some(x) => do yeet eyre!("Integer {value} followed by invalid token: {x}"),
        },

        Token::Semi => Statement::NullOpt,

        Token::Ident(ident) => {
            error!("Found ident {ident}. Identifier parsing not yet implemented.");
            Statement::NullOpt
        }
    };

    Ok(Some(statement))
}

fn parse(tokens: Vec<Token>) -> Result<AST> {
    let mut statements = Vec::new();
    let mut tokens = tokens.into_iter();

    loop {
        match get_next_statement(&mut tokens)? {
            None => break,
            Some(x) => statements.push(x),
        }
    }

    Ok(AST { statements })
}
