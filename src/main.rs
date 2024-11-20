#![feature(yeet_expr)]

use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{Read, Write},
    iter::Peekable,
};

use clap::Parser;
use common::use_recommended_logger;
use eyre::{eyre, Context, ContextCompat, Report, Result};
use log::{debug, info, warn};

#[derive(Parser)]
struct Cli {
    input: String,

    #[arg(short, long, global = true)]
    output: Option<String>,
}

#[derive(PartialEq, Clone, Debug)]
enum Token {
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
    Comma,
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
            Token::Comma => write!(f, ","),
            Token::Equal => write!(f, "="),
            Token::Semi => write!(f, ";"),
            Token::Let => write!(f, "the `let` keyword"),
            Token::Int(x) => write!(f, "an integer ({x})"),
            Token::Ident(x) => write!(f, "an identifier (\"{x}\")"),
            Token::EOF => write!(f, "the end of the file"),
        }
    }
}

#[derive(Debug)]
struct AST {
    statements: Vec<Statement>,
}

#[derive(Debug)]
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

#[derive(Debug, Clone)]
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
    Variable {
        name: String,
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
            Expression::Variable { name } => write!(f, "variable {name}"),
        }
    }
}

#[derive(Debug, Clone)]
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

trait Impossible<T> {
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
    debug!("{ast:?}");
    let mut binary = generate_assembly(ast).wrap_err("Failed to compile.")?;
    debug!("{binary:?}");

    File::create(output)
        .unwrap()
        .write_all(&mut binary)
        .unwrap();

    Ok(())
}

fn is_digit(c: &char) -> bool {
    c.is_digit(10)
}

fn is_normal_character(c: &char) -> bool {
    c.is_digit(36)
}

fn is_legal(c: &char) -> bool {
    ![' ', '\n', '\t'].contains(c)
}

struct SkipWhitespace<T>(Peekable<T>)
where
    T: Iterator<Item = char>;

impl<T> Iterator for SkipWhitespace<T>
where
    T: Iterator<Item = char>,
{
    type Item = <T as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next()? {
            ref x if !is_legal(x) => self.next(),
            x => Some(x),
        }
    }
}

impl<T> SkipWhitespace<T>
where
    T: Iterator<Item = char>,
{
    fn peek(&mut self) -> Option<<Self as Iterator>::Item> {
        while let Some(peek) = self.0.peek() {
            let peek = peek.clone();
            if is_legal(&peek) {
                return Some(peek);
            }
            self.0.next();
        }
        None
    }
}

fn build_string_from_init_and_while(c: &char, mut next: impl FnMut() -> Option<char>) -> String {
    let mut out = c.to_string();

    while let Some(next) = next() {
        out.push(next);
    }

    out
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut input = SkipWhitespace(input.chars().peekable());

    while let Some(next) = input.next() {
        let token = match next {
            '+' => Token::Add,
            '-' => Token::Sub,
            '*' => Token::Mul,
            '/' => Token::Div,
            '(' => Token::LParen,
            ')' => Token::RParen,
            ',' => Token::Comma,
            '=' => Token::Equal,
            ';' => Token::Semi,
            ref x if is_digit(x) => Token::Int(
                build_string_from_init_and_while(x, || match is_digit(&input.peek()?) {
                    true => input.next(),
                    false => None,
                })
                .parse()
                .unwrap(),
            ),
            ref x if is_normal_character(x) => {
                let string = build_string_from_init_and_while(x, || {
                    match is_normal_character(input.0.peek()?) {
                        true => input.next(),
                        false => None,
                    }
                });

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

fn remaining_expression_parsing(
    expression: Expression,
    tokens: &mut impl Iterator<Item = Token>,
    on_failed_parsing: impl FnOnce(Token, &mut dyn Iterator<Item = Token>) -> Result<Statement>,
) -> Result<Statement> {
    let statement = match tokens.next().impossible()? {
        Token::Semi | Token::EOF => Statement::Expression { value: expression },
        x @ (Token::Add | Token::Sub | Token::Mul | Token::Div) => {
            let expr = match get_next_statement(tokens)
                .wrap_err_with(|| {
                    format!("Failed to get right hand side of `{expression} {x} ...`")
                })?
                .impossible()?
            {
                Statement::Expression { value } => value,
                x => {
                    do yeet eyre!(
                        "Exprected right side of expression to be an expression, found {x}"
                    )
                }
            };

            Statement::Expression {
                value: Expression::MathematicalOperation {
                    left_side: Box::new(expression.clone()),
                    right_side: Box::new(expr.clone()),
                    operator: x.clone().try_into().wrap_err_with(|| {
                        format!(
                            "Failed to construct an expression out of `{expression} {x} {expr}`"
                        )
                    })?,
                },
            }
        }
        x => on_failed_parsing(x, tokens)?,
    };

    Ok(statement)
}

fn find_closing_brace(tokens: &mut dyn Iterator<Item = Token>) -> Result<Vec<Token>> {
    let mut tokens_in_parens = Vec::new();
    loop {
        let mut depth = 1;
        match tokens.next() {
            None => do yeet eyre!("Parenthesis was never closed."),
            Some(Token::LParen) => depth += 1,
            Some(Token::RParen) => depth -= 1,
            Some(x) => tokens_in_parens.push(x),
        }

        if depth == 0 {
            break;
        }
    }
    tokens_in_parens.push(Token::EOF);
    Ok(tokens_in_parens)
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

        Token::LParen => Statement::CodeBlock {
            ast: parse(
                find_closing_brace(tokens)
                    .wrap_err("Failed to parse tokens within a single opening parenthesis")?,
            )
            .wrap_err("Failed to parse tokens in parenthesis.")?,
        },

        x @ (Token::Mul | Token::Div | Token::Equal | Token::RParen | Token::Comma) => {
            do yeet eyre!("Detected junk: `{x}`")
        }

        Token::EOF => return Ok(None),

        Token::Int(value) => {
            remaining_expression_parsing(Expression::LiteralInt { value }, tokens, |token, _| {
                do yeet eyre!("Expression followed by invalid token: {token}")
            })
            .wrap_err_with(|| format!("Failed to parse the tokens after the integer {value}"))?
        }

        Token::Semi => Statement::NullOpt,

        Token::Ident(ident) => {
            let ident_clone = ident.clone();
            remaining_expression_parsing(
            Expression::Variable {
                name: ident.clone(),
            },
            tokens,
            |token, tokens| match token {
                Token::LParen => {
                    let mut args = Vec::new();
                    for item in find_closing_brace(tokens)
                        .wrap_err_with(|| {
                            format!(
                                "Failed to parse the arguments of the {ident}(...) function call."
                            )
                        })?
                        .split(|token| *token == Token::Comma)
                        .map(|token| {
                            parse(token.to_vec())
                                .wrap_err_with(|| format!("Failed to parse argument to {ident}."))
                        })
                    {
                            let ast = item?;
                            match ast.statements.len() {
                                0 => do yeet eyre!("Found to commas without an expression between them in function call {ident}."),
                                1 => {
                                    let Statement::Expression { ref value } = ast.statements[0] else {
                                        do yeet eyre!("Function parameters have to be statements, but one call of {ident} passes: {}", ast.statements[0])
                                    };
                                    args.push(value.clone());
                                }
                                2.. => do yeet eyre!("Found multiple statements not sperated by a comma in function call {ident}."),
                            }
                    }
                    Ok(Statement::Expression {
                        value: Expression::FunctionCall {
                            function_name: ident,
                            arguments: args,
                        },
                    })
                }
                x => do yeet eyre!("Invalid token after identifier {ident}: {x}"),
            },
        )
        .wrap_err_with(|| format!("Failed to parse the expression starting with {ident_clone}"))?
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
            Some(Statement::NullOpt) => {}
            Some(x) => statements.push(x),
        }
    }

    Ok(AST { statements })
}

#[derive(Debug)]
struct Function {
    arg_count: usize,
    binary: Vec<u8>,
    /// offset in .text, +0x400000 for real
    address: Option<i32>,
}

impl Function {
    fn write(&mut self, binary: &mut Vec<u8>) {
        self.address = Some(binary.len() as i32);
        binary.extend_from_slice(self.binary.as_slice());
    }
}

fn resolve_expression(
    value: Expression,
    text: &mut Vec<u8>,
    functions: &mut HashMap<String, Function>,
    variables: &mut HashMap<String, Expression>,
) -> Result<()> {
    match value {
        Expression::FunctionCall {
            function_name,
            arguments,
        } => {
            let Some(function) = functions.get_mut(function_name.as_str()) else {
                do yeet eyre!("No such function `{function_name}`.")
            };

            if function.arg_count != arguments.len() {
                do yeet eyre!(
                    "Invalid number of arguments for {function_name}: Expected {}, got {}",
                    function.arg_count,
                    arguments.len()
                )
            }

            if function.address == None {
                function.write(text);
            }

            let mut args = arguments.iter();

            if let Some(arg) = args.next() {
                store(text, Register::Rdi, arg, variables)?
            }

            let offset = function.address.impossible()? - text.len() as i32 + 5;
            // jmp
            text.push(0xE9);
            // relative address
            write_num(text, offset)
                .wrap_err_with(|| format!("Failed to store offset of {function:?}: {offset}"))?;
        }
        Expression::Variable { name } => {
            if variables.get(name.as_str()).is_none() {
                do yeet eyre!("Variable `{name}` not found.");
            }
        }
        Expression::MathematicalOperation {
            left_side,
            right_side,
            operator: _,
        } => {
            resolve_expression(*left_side, text, functions, variables)
                .wrap_err("Error within mathematical operation.")?;
            resolve_expression(*right_side, text, functions, variables)
                .wrap_err("Error within mathematical operation.")?;
        }
        _ => {}
    }
    Ok(())
}

enum Register {
    Eax,
    Ecx,
    Edx,
    Rdi,
}

fn write_num(binary: &mut Vec<u8>, value: i32) -> Result<()> {
    binary
        .write_all(&mut value.to_le_bytes())
        .wrap_err_with(|| format!("Failed to write {value} into output buffer."))
}

fn write_register_mov(binary: &mut Vec<u8>, register: Register) {
    let opcode = match register {
        Register::Eax => 0xB8,
        Register::Ecx => 0xB9,
        Register::Edx => 0xBA,
        Register::Rdi => todo!(),
    };
    binary.push(opcode);
}

fn store(
    binary: &mut Vec<u8>,
    register: Register,
    expression: &Expression,
    variables: &mut HashMap<String, Expression>,
) -> Result<()> {
    match expression {
        Expression::LiteralInt { value } => {
            write_register_mov(binary, register);
            write_num(binary, *value)?;
        }
        Expression::Variable { name } => {
            let expr = variables
                .get(name.as_str())
                .wrap_err_with(|| format!("Variable {name} not found."))?
                .clone();

            store(binary, register, &expr, variables)?
        }
        _ => todo!(),
    }

    Ok(())
}

fn generate_assembly(ast: AST) -> Result<Vec<u8>> {
    let mut text = Vec::new();
    let mut functions: HashMap<String, Function> = HashMap::new();
    let mut variables: HashMap<String, Expression> = HashMap::new();

    // 48 89 f7                mov    %rsi,%rdi
    // b8 01 00 00 00          mov    $0x1,%eax
    // bf 01 00 00 00          mov    $0x1,%edi
    // ba 05 00 00 00          mov    $0x5,%edx
    // c3                      ret
    let print = Function {
        arg_count: 1,
        binary: vec![
            0x48, 0x89, 0xf7, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xbf, 0x01, 0x0, 0x0, 0x0, 0xba, 0x05,
            0x0, 0x0, 0x0, 0xc3,
        ],
        address: None,
    };

    functions.insert("print".to_string(), print);

    for statement in ast.statements {
        match statement {
            Statement::Expression { value } => {
                resolve_expression(value, &mut text, &mut functions, &mut variables)?
            }
            Statement::VariableAssignment {
                variable_name,
                value,
            } => {
                variables.insert(variable_name, value);
            }
            _ => todo!(),
        }
    }

    Ok(text)
}
