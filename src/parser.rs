use crate::err;
use crate::Impossible;
use crate::Token;
use crate::UnlinkedTextSectionElement;

use std::fmt::Display;

use eyre::{eyre, Report, Result, WrapErr};

#[derive(PartialEq, Clone, Debug)]
pub struct Ast {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Code {
    Ast(Ast),
    Binary(Vec<UnlinkedTextSectionElement>),
}

#[derive(PartialEq, Clone, Debug)]
pub enum Statement {
    FunctionDefinition {
        function_name: String,
        parameters: Vec<String>,
        result: Expression,
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
                result: _,
            } => write!(f, "definition of function `{function_name}`"),
            Statement::NullOpt => write!(f, "empty statement"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
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
    Tuple {
        values: Vec<Expression>,
    },
    CodeBlock {
        ast: Ast,
    },
    Variable {
        name: String,
    },
    Unit,
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
            Expression::Tuple { values } => {
                let formatted = values
                    .iter()
                    .map(Expression::to_string)
                    .collect::<Vec<String>>()
                    .join(", ");

                write!(f, "({formatted})")
            }
            Expression::CodeBlock { ast } => write!(f, "a code block: {ast:?}"),
            Expression::Unit => write!(f, "the unit type"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Operator {
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
            x => Err(eyre!("Cannot convert {x} into an operator.")),
        }
    }
}

pub enum ExpressionParsingResult {
    Success { statement: Statement },
    Invalid { token: Token },
}

pub fn remaining_expression_parsing(
    expression: Expression,
    tokens: &mut impl Iterator<Item = Token>,
) -> Result<ExpressionParsingResult> {
    let statement = match tokens.next().impossible()? {
        Token::Semi | Token::Eof => Statement::Expression { value: expression },
        x @ (Token::Add | Token::Sub | Token::Mul | Token::Div) => {
            let expr = match get_next_statement(tokens)
                .wrap_err_with(|| {
                    format!("Failed to get right hand side of `{expression} {x} ...`")
                })?
                .impossible()?
            {
                Statement::Expression { value } => value,
                x => {
                    err!("Exprected right side of expression to be an expression, found {x}")
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
        token => return Ok(ExpressionParsingResult::Invalid { token }),
    };

    Ok(ExpressionParsingResult::Success { statement })
}

pub fn find_closing_brace(
    open: Token,
    close: Token,
    tokens: &mut dyn Iterator<Item = Token>,
) -> Result<Vec<Token>> {
    let mut tokens_in_parens = Vec::new();
    loop {
        let mut depth = 1;
        match tokens.next() {
            None => return Err(eyre!("Parenthesis was never closed.")),
            Some(t) if t == open => depth += 1,
            Some(t) if t == close => depth -= 1,
            Some(t) => tokens_in_parens.push(t),
        }

        if depth == 0 {
            break;
        }
    }
    tokens_in_parens.push(Token::Eof);
    Ok(tokens_in_parens)
}

pub fn get_next_statement(tokens: &mut impl Iterator<Item = Token>) -> Result<Option<Statement>> {
    let Some(next) = tokens.next() else {
        return Ok(None);
    };

    let statement = match next {
        Token::Let => {
            let variable_name = match tokens.next().unwrap() {
                Token::Ident(ident) => ident,
                x => return Err(eyre!("Expected identifier after let, found {x}")),
            };

            let next_token = tokens.next().unwrap();
            if next_token != Token::Equal {
                return Err(eyre!(
                    "Exprected equal sign (=) after `let {variable_name}`, found {next_token}"
                ));
            }

            let value = match get_next_statement(tokens) {
                Err(err) => Err(err).wrap_err_with(|| {
                    format!("Failed to get ... in `let {variable_name} = ...`")
                })?,
                Ok(None) => return Err(eyre!(
                    "Expected an expression to follow after `let {variable_name} = `, but got nothing."
                )),
                Ok(Some(Statement::Expression { value })) => value,
                Ok(Some(x)) => {
                    return Err(eyre!(
                    "Expected an expression to follow after `let {variable_name} = `, but got {x}."
                ))
                }
            };

            Statement::VariableAssignment {
                variable_name,
                value,
            }
        }

        Token::Fn => {
            let function_name = match tokens.next().unwrap() {
                Token::Ident(ident) => ident,
                x => return Err(eyre!("Expected identifier after let, found {x}")),
            };

            let next_token = tokens.next().unwrap();
            if next_token != Token::LParen {
                return Err(eyre!(
                    "Expected '(' after identifier in function definition, found {next_token}"
                ));
            }

            let mut parameters = Vec::new();
            loop {
                match tokens.next().unwrap() {
                    Token::RParen => break,
                    Token::Ident(next_token) => parameters.push(next_token),
                    other => return Err(eyre!("Invalid Token in function parameters: {other}")),
                };

                match tokens.next().unwrap() {
                    Token::Comma => continue,
                    Token::RParen => break,
                    token => {
                        return Err(eyre!(
                            "Invalid Token after function parameter identifier: {token}"
                        ));
                    }
                };
            }

            let code = match get_next_statement(tokens) {
                Ok(Some(statement)) => Ast {
                    statements: vec![statement],
                },

                Ok(None) => {
                    return Err(eyre!(
                        "Expected code block for function {function_name}, got nothing."
                    ))
                }
                Err(err) => Err(err).wrap_err(format!(
                    "Failed to parse code block for function `{function_name}`"
                ))?,
            };

            Statement::FunctionDefinition {
                function_name,
                parameters,
                result: Expression::CodeBlock { ast: code },
            }
        }

        // TODO: Remove duplication
        Token::Add => match get_next_statement(tokens) {
            Err(err) => {
                Err(err).wrap_err_with(|| "Failed to get the value after the `+`".to_string())?
            }
            Ok(None) => return Err(eyre!("Unexpected `+` with nothing after it.")),
            Ok(Some(Statement::Expression { value })) => Statement::Expression { value },
            Ok(Some(y)) => return Err(eyre!("Expected Expression after `+`, got {y}")),
        },
        Token::Sub => match get_next_statement(tokens) {
            Err(err) => {
                Err(err).wrap_err_with(|| "Failed to get the value after the `-`".to_string())?
            }
            Ok(None) => return Err(eyre!("Unexpected `-` with nothing after it.")),
            Ok(Some(Statement::Expression { value })) => Statement::Expression {
                value: Expression::MathematicalOperation {
                    left_side: Box::new(value),
                    right_side: Box::new(Expression::LiteralInt { value: -1 }),
                    operator: Operator::Mul,
                },
            },
            Ok(Some(y)) => return Err(eyre!("Expected Expression after `-`, got {y}")),
        },

        Token::LParen => {
            let values = find_closing_brace(Token::LParen, Token::RParen, tokens)
                .wrap_err("Failed to parse tokens withing parenthesis.")?
                .split(|x| *x == Token::Comma)
                .map(|x| Expression::CodeBlock {
                    ast: parse(x.to_vec()).unwrap(),
                })
                .collect();

            Statement::Expression {
                value: Expression::Tuple { values },
            }
        }

        Token::LCurly => {
            let block = find_closing_brace(Token::LCurly, Token::RCurly, tokens)
                .wrap_err("Failed to parse tokens within curly braces.")?;

            Statement::Expression {
                value: Expression::CodeBlock {
                    ast: parse(block).wrap_err("Failed to parse tokens in parenthesis.")?,
                },
            }
        }

        x @ (Token::Mul
        | Token::Div
        | Token::Equal
        | Token::RParen
        | Token::RCurly
        | Token::Comma) => return Err(eyre!("Detected junk: `{x}`")),

        Token::Eof => return Ok(None),

        Token::Int(value) => {
            match remaining_expression_parsing(Expression::LiteralInt { value }, tokens) {
                Err(err) => Err(err).wrap_err_with(|| {
                    format!("Failed to parse the tokens after the integer {value}")
                })?,
                Ok(ExpressionParsingResult::Invalid { token }) => {
                    Err(eyre!("Expression followed by invalid token: {token}"))?
                }
                Ok(ExpressionParsingResult::Success { statement }) => statement,
            }
        }

        Token::Semi => Statement::NullOpt,

        Token::Ident(ident) => {
            let res = remaining_expression_parsing(
                Expression::Variable {
                    name: ident.clone(),
                },
                tokens,
            );

            match res
                .wrap_err_with(|| format!("Failed to parse the expression starting with {ident}"))?
            {
                ExpressionParsingResult::Success { statement } => statement,
                ExpressionParsingResult::Invalid { token } => match token {
                    Token::LParen => {
                        let mut args = Vec::new();
                        let content = find_closing_brace(Token::LParen, Token::RParen, tokens)
                            .wrap_err_with(|| {
                                format!("Failed to parse arguments of {ident}(...)")
                            })?;

                        let tokens = content.split(|token| *token == Token::Comma);

                        for item in tokens {
                            let ast = parse(item.to_vec())?;

                            let mut statements = ast.statements.iter();

                            while let Some(Statement::Expression { value }) = statements.next() {
                                args.push(value.clone());
                            }
                        }

                        Statement::Expression {
                            value: Expression::FunctionCall {
                                function_name: ident,
                                arguments: args,
                            },
                        }
                    }
                    x => err!("Invalid token after identifier {ident}: {x}"),
                },
            }
        }
    };

    Ok(Some(statement))
}

pub fn parse(tokens: Vec<Token>) -> Result<Ast> {
    let mut statements = Vec::new();
    let mut tokens = tokens.into_iter();

    loop {
        match get_next_statement(&mut tokens)? {
            None => break,
            Some(Statement::NullOpt) => {}
            Some(x) => statements.push(x),
        }
    }

    Ok(Ast { statements })
}
