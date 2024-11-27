#![allow(dead_code)]
#![feature(yeet_expr)]

use std::{
    collections::HashMap,
    fmt::Display,
    fs::{metadata, set_permissions, File},
    io::{Read, Write},
    iter::{repeat, Peekable},
    os::unix::fs::PermissionsExt,
    slice,
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

    // Tokenize
    let tokens = tokenize(&content).wrap_err("Failed to tokenize.")?;
    debug!("Tokenizer Output: {tokens:?}");

    // Parse
    let ast = parse(tokens).wrap_err("Failed to parse.")?;
    debug!("Parser Output: {ast:?}");

    // Compile
    let mut main = Function {
        code: ast,
        address: None,
        symbols: HashMap::new(),
    };
    let mut program = Program::new();
    main.write(&mut program.text)?;
    debug!("Compiler output: {program:?}");
    let mut elf = program.generate_elf();

    File::create(&output).unwrap().write_all(&mut elf).unwrap();
    let mut permissions = metadata(&output).unwrap().permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    set_permissions(output, permissions).unwrap();
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

fn extend<A: Clone, B>(vec: &mut Vec<A>, b: B) {
    unsafe {
        vec.extend_from_slice(slice::from_raw_parts(
            &b as *const B as *const A,
            size_of::<B>(),
        ))
    }
}

impl Program {
    fn write_all(&mut self, binary: &[u8]) {
        self.text
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
    code_pre: Vec<Executable>,
    returns: Loadable,
    code_post: Vec<Executable>,
}

impl Load {
    fn simple(loadable: Loadable) -> Self {
        Self {
            code_pre: vec![],
            returns: loadable,
            code_post: vec![],
        }
    }
}

#[derive(Clone, Debug)]
enum Executable {
    Push(Register),
    Pop(Register),
    Call(i32),
    MoveLoad { src: Loadable, dest: Loadable },
    Syscall,
    Ret,
}

#[derive(Clone, Debug)]
enum Loadable {
    Register(Register),
    Work(Vec<u8>, Box<Loadable>),
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
    symbols: HashMap<String, Loadable>,
}

#[derive(Clone, Debug)]
struct Program {
    symbols: HashMap<String, Symbol>,
    text: Vec<u8>,
    data: Vec<u8>,
}

impl Program {
    fn new() -> Self {
        Program {
            symbols: HashMap::new(),
            text: Vec::new(),
            data: b"test".to_vec(),
        }
    }
}

fn sys_v_calling_convention() -> impl Iterator<Item = Loadable> {
    vec![
        Loadable::Register(Register::Rdi),
        Loadable::Register(Register::Rsi),
        Loadable::Register(Register::Rdx),
        Loadable::Register(Register::Rcx),
        Loadable::Register(Register::R8),
        Loadable::Register(Register::R9),
    ]
    .into_iter()
    .chain(repeat(Loadable::Stack))
}

fn load_all(parameters: Vec<String>) -> HashMap<String, Loadable> {
    parameters
        .into_iter()
        .zip(sys_v_calling_convention())
        .collect()
}

fn move_all(parameters: Vec<Load>) -> Vec<Executable> {
    parameters
        .into_iter()
        .zip(sys_v_calling_convention())
        .map(|(mut src, dest)| {
            let mut vec = src.code_pre;
            vec.push(Executable::MoveLoad {
                src: src.returns,
                dest,
            });
            vec.append(&mut src.code_post);
            vec
        })
        .flatten()
        .collect()
}

fn load(expression: &Expression, program_context: &mut Program) -> Result<Load> {
    let load = match expression {
        Expression::LiteralInt { value } => Load::simple(Loadable::Immediate(*value)),
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
    program_context: &mut Program,
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
    code_pre.push(Executable::Push(Register::Rax));
    code_pre.push(Executable::Call(relative));

    Ok(Load {
        code_pre,
        returns: Loadable::Register(Register::Rax),
        code_post: vec![Executable::Pop(Register::Rax)],
    })
}

fn resolve_standalone_expression(
    value: &Expression,
    program_context: &mut Program,
) -> Result<Vec<Executable>> {
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
impl Program {
    fn assembly_function(&mut self, executable: &Vec<Executable>) -> Function {
        self.binary_function(self.as_binary(executable))
    }

    fn binary_function(&mut self, mut binary: Vec<u8>) -> Function {
        let function = Function::dummy(&self.text);
        self.text.append(&mut binary);
        function
    }

    fn load_onto_stack(&self, loadable: &Loadable) -> Vec<u8> {
        match loadable {
            Loadable::Stack => Vec::new(),
            Loadable::Register(x) => self.one_as_binary(&Executable::Push(*x)),
            Loadable::Immediate(x) => self.one_as_binary(&Executable::MoveLoad {
                src: Loadable::Immediate(*x),
                dest: Loadable::Register(Register::Rsp),
            }),
            Loadable::Work(first, then) => {
                let mut vec = first.clone();
                vec.append(&mut self.one_as_binary(&Executable::MoveLoad {
                    src: *then.clone(),
                    dest: Loadable::Register(Register::Rsp),
                }));
                vec
            }
        }
    }

    fn load_into_register(&self, loadable: &Loadable, dest: &Register) -> Vec<u8> {
        match loadable {
            Loadable::Stack => self.one_as_binary(&Executable::Pop(*dest)),
            Loadable::Register(src) => {
                // MOV r64, r/m64
                // REX.W + 8B /r
                vec![
                    Rex::registers_64(src, dest),
                    0x8B,
                    ModRM::register(dest, src),
                ]
            }
            Loadable::Work(first, then) => {
                let mut vec = first.clone();
                vec.append(&mut self.one_as_binary(&Executable::MoveLoad {
                    src: *then.clone(),
                    dest: Loadable::Register(*dest),
                }));
                vec
            }
            Loadable::Immediate(x) => {
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

    fn one_as_binary(&self, executable: &Executable) -> Vec<u8> {
        match executable {
            // PUSH r64
            // 50 + rd
            Executable::Push(register) => match extended(register) {
                false => vec![0x50 + *register as u8],
                true => vec![0x41, *register as u8 - 8],
            },
            // POP r64
            // 58 + rd
            Executable::Pop(register) => match extended(register) {
                false => vec![0x58 + *register as u8],
                true => vec![0x41, *register as u8],
            },
            Executable::MoveLoad { src, dest } => match (src, dest) {
                (_, Loadable::Immediate(_)) => {
                    panic!("Cannot move a value into an immediate.")
                }
                (_, Loadable::Work(_, _)) => {
                    panic!("Cannot load into work.")
                }
                (loadable, Loadable::Stack) => self.load_onto_stack(loadable),
                (loadable, Loadable::Register(dest)) => self.load_into_register(loadable, dest),
            },
            // CALL rel32
            Executable::Call(relative_offset) => vec![0xE8]
                .into_iter()
                .chain(relative_offset.to_le_bytes())
                .collect(),
            // SYSCALL
            // 0F 05
            Executable::Syscall => {
                vec![0x0f, 0x05]
            }
            // RET
            // C3
            Executable::Ret => {
                vec![0xC3]
            }
        }
    }

    fn as_binary(&self, executable: &Vec<Executable>) -> Vec<u8> {
        let mut binary = Vec::new();
        for executable in executable {
            binary.append(&mut self.one_as_binary(executable));
        }
        binary
    }
}

fn basic_functions(program_context: &mut Program) -> Result<()> {
    let putchar = program_context.assembly_function(&vec![
        Executable::MoveLoad {
            src: Loadable::Register(Register::Rdi),
            dest: Loadable::Register(Register::Rsi),
        },
        Executable::MoveLoad {
            src: Loadable::Immediate(1),
            dest: Loadable::Register(Register::Rax),
        },
        Executable::MoveLoad {
            src: Loadable::Immediate(1),
            dest: Loadable::Register(Register::Rdi),
        },
        Executable::MoveLoad {
            src: Loadable::Immediate(1),
            dest: Loadable::Register(Register::Rdx),
        },
        Executable::Syscall,
        Executable::Ret,
    ]);

    let exit = program_context.assembly_function(&vec![
        Executable::MoveLoad {
            src: Loadable::Immediate(60),
            dest: Loadable::Register(Register::Rax),
        },
        Executable::MoveLoad {
            src: Loadable::Immediate(0),
            dest: Loadable::Register(Register::Rdi),
        },
        Executable::Syscall,
        Executable::Ret,
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
    fn dummy(text: &Vec<u8>) -> Self {
        Function {
            code: AST {
                statements: Vec::new(),
            },
            address: Some(text.len() as u64),
            symbols: HashMap::new(),
        }
    }

    fn get_relative_jump(&mut self, text: &mut Vec<u8>) -> Result<i32> {
        let address = match self.address {
            Some(address) => address,
            None => {
                self.write(text).wrap_err("Failed to create self.")?;
                self.address.impossible().unwrap()
            }
        };

        Ok(text.len() as i32 - address as i32 + 5)
    }

    fn write(&mut self, program: &mut Vec<u8>) -> Result<()> {
        self.address = Some(program.len() as u64);
        program.write_all(&self.compile()?).unwrap();
        Ok(())
    }

    fn compile(&self) -> Result<Vec<u8>> {
        let mut program_context = Program::new();
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
            let executable = match statement {
                Statement::Expression { value } => {
                    resolve_standalone_expression(&value, &mut program_context)
                        .wrap_err("Failed to resolve standalone expression.")?
                }
                _ => todo!(),
            };
            debug!("Generated high level assembly: {executable:?}");
            program_context
                .text
                .write_all(&mut program_context.as_binary(&executable))
                .unwrap();
        }

        Ok(program_context.text)
    }
}

const PAGE_SIZE: u64 = 0x1000;
const SECTION_ALIGNMENT: u64 = PAGE_SIZE;

struct Aligned {
    required_padding: u64,
    padding: Vec<u8>,
}

fn align(number: u64, alignment: u64) -> Aligned {
    let required_padding = alignment - number % alignment;
    Aligned {
        required_padding,
        padding: vec![0; required_padding as usize],
    }
}

fn align_vector(vec: &mut Vec<u8>, alignment: u64) -> u64 {
    vec.append(&mut align(vec.len() as u64, alignment).padding);
    vec.len() as u64
}

fn align_program_header_offset(p_vaddr: u64, offset: u64) -> Aligned {
    let required =
        dbg!(PAGE_SIZE + dbg!(p_vaddr % PAGE_SIZE) - dbg!(offset % PAGE_SIZE)) % PAGE_SIZE;
    Aligned {
        required_padding: required,
        padding: vec![0; required as usize],
    }
}

impl Program {
    fn generate_elf(mut self) -> Vec<u8> {
        let mut p_vaddr = 0x400000;

        let mut vec = Vec::new();

        let bin = |slice: &[&str]| {
            slice
                .iter()
                .map(|x| x.as_bytes().to_vec())
                .flatten()
                .collect::<Vec<u8>>()
        };

        let sections = [".text\0", ".data\0", ".shstrtab\0"];
        let mut sections_bin = bin(&sections);
        let mut n = 0;
        let mut next_section = || {
            let res = bin(&sections[0..n as usize]).len() as u32;
            n += 1;
            res
        };

        let text_len = dbg!(align_vector(&mut self.text, SECTION_ALIGNMENT));
        let data_len = align_vector(&mut self.data, SECTION_ALIGNMENT);

        let elf_header_size = size_of::<Elf64Header>() as u64;
        let elf_program_header_size = size_of::<Elf64ProgramHeader>() as u64;
        let elf_section_header_size = size_of::<Elf64SectionHeader>() as u64;

        let headers = elf_header_size + 2 * elf_program_header_size + 3 * elf_section_header_size;
        let mut header_padding = align(headers, PAGE_SIZE);
        let header_len = headers + header_padding.required_padding;

        let elf_header = Elf64Header {
            e_ident: [0x7F, b'E', b'L', b'F', 2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            e_type: 2,
            e_machine: 0x3E,
            e_version: 1,
            e_entry: p_vaddr,
            e_phoff: elf_header_size,
            e_shoff: elf_header_size + 2 * elf_program_header_size,
            e_flags: 0,
            e_ehsize: elf_header_size as u16,
            e_phentsize: elf_program_header_size as u16,
            e_phnum: 2,
            e_shentsize: elf_section_header_size as u16,
            e_shnum: 3,
            e_shstrndx: 2,
        };

        let text_program_header = Elf64ProgramHeader {
            p_type: 1,
            p_flags: 1 | 4,
            p_offset: header_len,
            p_vaddr,
            p_paddr: p_vaddr,
            p_filesz: text_len,
            p_memsz: text_len,
            p_align: PAGE_SIZE,
        };
        dbg!(&text_program_header).assert_valid();
        p_vaddr += text_len;

        let data_program_header = Elf64ProgramHeader {
            p_type: 1,
            p_flags: 2 | 4,
            p_offset: header_len + text_len,
            p_vaddr,
            p_paddr: p_vaddr,
            p_filesz: data_len,
            p_memsz: data_len,
            p_align: PAGE_SIZE,
        };
        dbg!(&data_program_header).assert_valid();

        let text_section_header = Elf64SectionHeader {
            sh_name: next_section(),
            sh_type: 1,
            sh_flags: 2 | 4,
            sh_addr: 0x400000,
            sh_offset: header_len,
            sh_size: text_len,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: SECTION_ALIGNMENT,
            sh_entsize: 0,
        };
        let data_section_header = Elf64SectionHeader {
            sh_name: next_section(),
            sh_type: 1,
            sh_flags: 1 | 2,
            sh_addr: 0x400000 + text_section_header.sh_size,
            sh_offset: header_len + text_len,
            sh_size: data_len,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: SECTION_ALIGNMENT,
            sh_entsize: 0,
        };
        let strings_section_header = Elf64SectionHeader {
            sh_name: next_section(),
            sh_type: 3,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: header_len + text_len + data_len,
            sh_size: sections_bin.len() as u64,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 0x0,
            sh_entsize: 0,
        };

        extend(&mut vec, elf_header);
        extend(&mut vec, text_program_header);
        extend(&mut vec, data_program_header);
        extend(&mut vec, text_section_header);
        extend(&mut vec, data_section_header);
        extend(&mut vec, strings_section_header);

        vec.append(&mut header_padding.padding);
        vec.append(&mut self.text);
        vec.append(&mut self.data);
        vec.append(&mut sections_bin);

        vec
    }
}

#[repr(C)]
struct Elf64Header {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

#[derive(Debug)]
#[repr(C)]
struct Elf64ProgramHeader {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

impl Elf64ProgramHeader {
    fn assert_valid(&self) {
        assert_eq!((self.p_vaddr - self.p_offset) % PAGE_SIZE, 0);
        assert_eq!((self.p_paddr - self.p_offset) % PAGE_SIZE, 0);
    }
}

#[repr(C)]
struct Elf64SectionHeader {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}
