use crate::Impossible;
use crate::Token;

use std::fmt::Display;

use eyre::{eyre, Report, Result, WrapErr};

#[derive(Clone, Debug)]
pub struct AST {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug)]
pub enum Statement {
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
            x => do yeet eyre!("Cannot convert {x} into an operator."),
        }
    }
}

pub fn remaining_expression_parsing(
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

pub fn find_closing_brace(tokens: &mut dyn Iterator<Item = Token>) -> Result<Vec<Token>> {
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

pub fn get_next_statement<'a>(
    tokens: &mut impl Iterator<Item = Token>,
) -> Result<Option<Statement>> {
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

pub fn parse(tokens: Vec<Token>) -> Result<AST> {
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
