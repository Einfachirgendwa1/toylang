#![feature(yeet_expr)]

use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{Read, Write},
    iter::{repeat, Peekable},
};

use clap::Parser;
use common::use_recommended_logger;
use eyre::{eyre, Context, ContextCompat, Report, Result};
use log::{debug, error, info, warn};

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
    Int(i64),
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

#[derive(Clone, Debug)]
struct AST {
    statements: Vec<Statement>,
}

#[derive(Clone, Debug)]
enum Statement {
    FunctionDefinition {
        function_name: String,
        parameters: Vec<String>,
        code: AST,
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
                parameters: _,
                code: _,
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
        value: i64,
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
    debug!("Tokenizer Output: {tokens:?}");
    let ast = parse(tokens).wrap_err("Failed to parse.")?;
    debug!("Parser Output: {ast:?}");

    let mut main = Function {
        code: ast,
        address: None,
        symbols: HashMap::new(),
    };
    let mut text = TextSection(Vec::new());
    main.write(&mut text)?;
    debug!("Compiler output: {text:?}");

    File::create(output)
        .unwrap()
        .write_all(&mut text.0)
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
            x => Err(eyre!(
                "Found an invalid character: {x} (character code {})",
                x as u8
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

type Address = u64;

#[derive(Clone, Debug)]
struct TextSection(Vec<u8>);

impl TextSection {
    fn head(&self) -> Address {
        self.0.len() as Address
    }

    fn write_all(&mut self, binary: &[u8]) {
        self.0
            .write_all(&binary)
            .expect("Failed to write to buffer.")
    }
}

#[derive(Clone, Copy, Debug)]
enum Register {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
}

impl Register {
    const fn byte(&self) -> u8 {
        match self {
            Register::Rax => 0,
            Register::Rcx => 1,
            Register::Rdx => 2,
            Register::Rbx => 3,
            Register::Rsp => 4,
            Register::Rbp => 5,
            Register::Rsi => 6,
            Register::Rdi => 7,
            Register::R8 => 0x58,
            Register::R9 => 0x59,
        }
    }
}

#[derive(Clone, Debug)]
struct Load {
    code_pre: Vec<Executeable>,
    returns: Loadeable,
    code_post: Vec<Executeable>,
}

impl Load {
    fn simple(loadeable: Loadeable) -> Self {
        Self {
            code_pre: vec![],
            returns: loadeable,
            code_post: vec![],
        }
    }
}

#[derive(Clone, Debug)]
enum Executeable {
    Push(Register),
    Pop(Register),
    Call(i32),
    MoveLoad { src: Loadeable, dest: Loadeable },
    Syscall,
    Ret,
}

#[derive(Clone, Debug)]
enum Loadeable {
    Register(Register),
    Work(Vec<u8>, Box<Loadeable>),
    Immediate(i64),
    Stack,
}

#[derive(Clone, Debug)]
enum Symbol {
    Variable(Expression),
    Function(Function),
}

#[derive(Clone, Debug)]
struct Function {
    code: AST,
    address: Option<Address>,
    symbols: HashMap<String, Loadeable>,
}

#[derive(Clone, Debug)]
struct ProgramContext {
    symbols: HashMap<String, Symbol>,
    text: TextSection,
}

fn sys_v_calling_convention() -> impl Iterator<Item = Loadeable> {
    vec![
        Loadeable::Register(Register::Rdi),
        Loadeable::Register(Register::Rsi),
        Loadeable::Register(Register::Rdx),
        Loadeable::Register(Register::Rcx),
        Loadeable::Register(Register::R8),
        Loadeable::Register(Register::R9),
    ]
    .into_iter()
    .chain(repeat(Loadeable::Stack))
}

fn load_all(parameters: Vec<String>) -> HashMap<String, Loadeable> {
    parameters
        .into_iter()
        .zip(sys_v_calling_convention())
        .collect()
}

fn move_all(parameters: Vec<Load>) -> Vec<Executeable> {
    parameters
        .into_iter()
        .zip(sys_v_calling_convention())
        .map(|(mut src, dest)| {
            let mut vec = src.code_pre;
            vec.push(Executeable::MoveLoad {
                src: src.returns,
                dest,
            });
            vec.append(&mut src.code_post);
            vec
        })
        .flatten()
        .collect()
}

fn load(expression: &Expression, program_context: &mut ProgramContext) -> Result<Load> {
    let load = match expression {
        Expression::LiteralInt { value } => Load::simple(Loadeable::Immediate(*value)),
        Expression::Variable { name } => {
            let Symbol::Variable(_var) = program_context
                .symbols
                .get(name)
                .with_context(|| format!("Variable {name} doesn't exist!"))?
                .clone()
            else {
                do yeet eyre!("Expected symbol {name} to be a variable, but it's not.");
            };

            todo!()
        }

        Expression::FunctionCall {
            function_name,
            arguments,
        } => build_function(&function_name, arguments, program_context)?,
        _ => todo!(),
    };

    Ok(load)
}

fn build_function(
    name: &str,
    args: &Vec<Expression>,
    program_context: &mut ProgramContext,
) -> Result<Load> {
    let Symbol::Function(function) = program_context
        .symbols
        .get_mut(name)
        .wrap_err_with(|| format!("Function {name} doesn't exist."))?
    else {
        do yeet eyre!("{name} is not a function.");
    };

    let relative = function.get_relative_jump(&mut program_context.text)?;

    let mut loads = Vec::new();
    for arg in args {
        loads.push(load(arg, program_context).wrap_err("In function parameter.")?);
    }

    let mut code_pre = move_all(loads);
    code_pre.push(Executeable::Push(Register::Rax));
    code_pre.push(Executeable::Call(relative));

    Ok(Load {
        code_pre,
        returns: Loadeable::Register(Register::Rax),
        code_post: vec![Executeable::Pop(Register::Rax)],
    })
}

fn resolve_standalone_expression(
    value: &Expression,
    program_context: &mut ProgramContext,
) -> Result<Vec<Executeable>> {
    match value {
        Expression::FunctionCall {
            function_name,
            arguments: args,
        } => {
            let mut load = build_function(&function_name, args, program_context)?;
            let mut vec = load.code_pre;
            vec.append(&mut load.code_post);
            Ok(vec)
        }
        Expression::MathematicalOperation {
            left_side,
            right_side,
            operator: _,
        } => {
            let mut vec = resolve_standalone_expression(&left_side, program_context)?;
            let mut rhs = resolve_standalone_expression(&right_side, program_context)?;
            vec.append(&mut rhs);
            Ok(vec)
        }

        expr => {
            warn!("Unused expression {expr}");
            Ok(Vec::new())
        }
    }
}

const fn extended(register: &Register) -> bool {
    register.byte() >= 8
}

// Source for the assembly: https://www.felixcloutier.com/x86
impl ProgramContext {
    fn assembly_function(&mut self, executeable: &Vec<Executeable>) -> Function {
        self.binary_function(self.as_binary(executeable))
    }

    fn binary_function(&mut self, mut binary: Vec<u8>) -> Function {
        let function = Function::dummy(&self.text);
        self.text.0.append(&mut binary);
        function
    }

    fn load_onto_stack(&self, loadeable: &Loadeable) -> Vec<u8> {
        match loadeable {
            Loadeable::Stack => Vec::new(),
            Loadeable::Register(x) => self.one_as_binary(&Executeable::Push(*x)),
            Loadeable::Immediate(x) => self.one_as_binary(&Executeable::MoveLoad {
                src: Loadeable::Immediate(*x),
                dest: Loadeable::Register(Register::Rsp),
            }),
            Loadeable::Work(first, then) => {
                let mut vec = first.clone();
                vec.append(&mut self.one_as_binary(&Executeable::MoveLoad {
                    src: *then.clone(),
                    dest: Loadeable::Register(Register::Rsp),
                }));
                vec
            }
        }
    }

    fn load_into_register(&self, loadeable: &Loadeable, dest: &Register) -> Vec<u8> {
        match loadeable {
            Loadeable::Stack => self.one_as_binary(&Executeable::Pop(*dest)),
            Loadeable::Register(src) => {
                // MOV r64, r/m64
                // REX.W + 8B /r
                vec![
                    Rex::registers_64(src, dest),
                    0x8B,
                    ModRM::register(dest, src),
                ]
            }
            Loadeable::Work(first, then) => {
                let mut vec = first.clone();
                vec.append(&mut self.one_as_binary(&Executeable::MoveLoad {
                    src: *then.clone(),
                    dest: Loadeable::Register(*dest),
                }));
                vec
            }
            Loadeable::Immediate(x) => {
                // MOV r64, imm64
                // REX.W, B8 + rd, io
                let mut vec = vec![
                    Rex::byte(true, extended(dest), false, false),
                    0xB8 + dest.byte(),
                ];
                vec.extend_from_slice(&x.to_le_bytes());
                vec
            }
        }
    }

    fn one_as_binary(&self, executeable: &Executeable) -> Vec<u8> {
        match executeable {
            // PUSH r64
            // 50 + rd
            Executeable::Push(register) => match extended(register) {
                false => vec![0x50 + *register as u8],
                true => vec![0x41, *register as u8 - 8],
            },
            // POP r64
            // 58 + rd
            Executeable::Pop(register) => match extended(register) {
                false => vec![0x58 + *register as u8],
                true => vec![0x41, *register as u8],
            },
            Executeable::MoveLoad { src, dest } => match (src, dest) {
                (_, Loadeable::Immediate(_)) => {
                    panic!("Cannot move a value into an immediate.")
                }
                (_, Loadeable::Work(_, _)) => {
                    panic!("Cannot load into work.")
                }
                (loadeable, Loadeable::Stack) => self.load_onto_stack(loadeable),
                (loadeable, Loadeable::Register(dest)) => self.load_into_register(loadeable, dest),
            },
            // CALL rel32
            Executeable::Call(relative_offset) => vec![0xE8]
                .into_iter()
                .chain(relative_offset.to_le_bytes())
                .collect(),
            // SYSCALL
            // 0F 05
            Executeable::Syscall => {
                vec![0x0f, 0x05]
            }
            // RET
            // C3
            Executeable::Ret => {
                vec![0xC3]
            }
        }
    }

    fn as_binary(&self, executeable: &Vec<Executeable>) -> Vec<u8> {
        let mut binary = Vec::new();
        for executeable in executeable {
            binary.append(&mut self.one_as_binary(executeable));
        }
        binary
    }
}

fn basic_functions(program_context: &mut ProgramContext) -> Result<()> {
    let putchar = program_context.assembly_function(&vec![
        Executeable::MoveLoad {
            src: Loadeable::Register(Register::Rdi),
            dest: Loadeable::Register(Register::Rsi),
        },
        Executeable::MoveLoad {
            src: Loadeable::Immediate(1),
            dest: Loadeable::Register(Register::Rax),
        },
        Executeable::MoveLoad {
            src: Loadeable::Immediate(1),
            dest: Loadeable::Register(Register::Rdi),
        },
        Executeable::MoveLoad {
            src: Loadeable::Immediate(1),
            dest: Loadeable::Register(Register::Rdx),
        },
        Executeable::Syscall,
        Executeable::Ret,
    ]);

    let exit = program_context.assembly_function(&vec![
        Executeable::MoveLoad {
            src: Loadeable::Immediate(60),
            dest: Loadeable::Register(Register::Rax),
        },
        Executeable::MoveLoad {
            src: Loadeable::Immediate(0),
            dest: Loadeable::Register(Register::Rdi),
        },
        Executeable::Syscall,
        Executeable::Ret,
    ]);

    program_context
        .symbols
        .insert("putchar".to_string(), Symbol::Function(putchar));

    program_context
        .symbols
        .insert("exit".to_string(), Symbol::Function(exit));

    Ok(())
}

struct Rex {
    w: bool,
    r: bool,
    x: bool,
    b: bool,
}

impl Rex {
    const fn registers_64(src: &Register, dest: &Register) -> u8 {
        Rex::byte(true, extended(dest), false, extended(src))
    }

    const fn byte(w: bool, r: bool, x: bool, b: bool) -> u8 {
        Rex { w, r, x, b }.prefix()
    }

    const fn prefix(&self) -> u8 {
        let mut val = 0b01000000u8;
        if self.w {
            val |= 0b1000;
        }
        if self.r {
            val |= 0b100;
        }
        if self.x {
            val |= 0b10;
        }
        if self.b {
            val |= 0b1;
        }

        val
    }
}

struct ModRM {
    mod_bits: Mod,
    reg_or_opcode: u8,
    r_or_m: u8,
}

enum Mod {
    AddrNoDisplacement,
    Addr8BitDisplacement,
    Addr32BitDisplacement,
    Register,
}

impl Mod {
    const fn byte(&self) -> u8 {
        match self {
            Mod::AddrNoDisplacement => 0b00,
            Mod::Addr8BitDisplacement => 0b01,
            Mod::Addr32BitDisplacement => 0b10,
            Mod::Register => 0b11,
        }
    }
}

impl ModRM {
    const fn register(dest: &Register, src: &Register) -> u8 {
        assert!(!extended(src));
        assert!(!extended(dest));
        ModRM {
            mod_bits: Mod::Register,
            reg_or_opcode: *dest as u8,
            r_or_m: *src as u8,
        }
        .byte()
    }

    const fn byte(&self) -> u8 {
        self.mod_bits.byte() << 6 | self.reg_or_opcode << 3 | self.r_or_m
    }
}

impl Function {
    fn dummy(text: &TextSection) -> Self {
        Function {
            code: AST {
                statements: Vec::new(),
            },
            address: Some(text.head()),
            symbols: HashMap::new(),
        }
    }

    fn get_relative_jump(&mut self, text: &mut TextSection) -> Result<i32> {
        let address = match self.address {
            Some(address) => address,
            None => {
                self.write(text).wrap_err("Failed to create self.")?;
                self.address.impossible().unwrap()
            }
        };

        Ok(text.head() as i32 - address as i32 + 5)
    }

    fn write(&mut self, text: &mut TextSection) -> Result<()> {
        self.address = Some(text.head());
        text.write_all(&self.compile()?);
        Ok(())
    }

    fn compile(&self) -> Result<Vec<u8>> {
        let mut program_context = ProgramContext {
            symbols: HashMap::new(),
            text: TextSection(Vec::new()),
        };
        let mut functions_within = HashMap::new();

        basic_functions(&mut program_context)?;

        let mut statements = self.code.statements.clone();
        statements.push(Statement::Expression {
            value: Expression::FunctionCall {
                function_name: "exit".to_string(),
                arguments: vec![],
            },
        });

        for statement in &statements {
            if let Statement::FunctionDefinition {
                function_name,
                parameters,
                code,
            } = statement
            {
                let function = Function {
                    code: code.clone(),
                    address: None,
                    symbols: load_all(parameters.clone()),
                };
                functions_within.insert(function_name.clone(), function);
            }
        }

        for statement in statements {
            let executeable = match statement {
                Statement::Expression { value } => {
                    resolve_standalone_expression(&value, &mut program_context)
                        .wrap_err("Failed to resolve standalone expression.")?
                }
                _ => todo!(),
            };
            debug!("Generated high level assembly: {executeable:?}");
            program_context
                .text
                .0
                .write_all(&mut program_context.as_binary(&executeable))
                .unwrap();
        }

        Ok(program_context.text.0)
    }
}
