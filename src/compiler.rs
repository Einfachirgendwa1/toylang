use crate::{elf::generate_elf, link, Ast, Code, Expression, Statement};

use std::{collections::HashMap, iter::repeat};

use eyre::{eyre, ContextCompat, Result, WrapErr};
use log::{debug, warn};

#[derive(Clone, Copy, Debug)]
pub enum Register {
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
    MoveLoad { src: Loadable, dest: Loadable },
    Call { label: String },
    Syscall,
    Ret,
}

#[derive(Clone, Debug)]
pub enum Loadable {
    Register(Register),
    Work(Vec<UnlinkedTextSectionElement>, Box<Loadable>),
    Immediate(i64),
    Stack,
}

#[derive(Clone, Debug)]
pub enum Symbol {
    Variable(Expression),
    Function { function: Function, written: bool },
}

#[derive(Clone, Debug)]
pub struct Function {
    pub ident: String,
    pub code: Code,
    pub symbols: HashMap<String, Loadable>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub symbols: HashMap<String, Symbol>,
    pub text: Vec<Label>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Label {
    pub ident: String,
    pub code: Vec<UnlinkedTextSectionElement>,
}

#[derive(Clone, Debug)]
pub enum UnlinkedTextSectionElement {
    Binary(Vec<u8>),
    Call { function_name: String },
}

impl Program {
    pub fn new() -> Self {
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
        .flat_map(|(mut src, dest)| {
            let mut vec = src.code_pre;
            vec.push(Executable::MoveLoad {
                src: src.returns,
                dest,
            });
            vec.append(&mut src.code_post);
            vec
        })
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
                return Err(eyre!(
                    "Expected symbol {name} to be a variable, but it's not."
                ));
            };

            todo!()
        }

        Expression::FunctionCall {
            function_name,
            arguments,
        } => build_function(function_name, arguments, program_context)?,
        _ => todo!(),
    };

    Ok(load)
}

fn build_function(
    name: &str,
    args: &Vec<Expression>,
    program_context: &mut Program,
) -> Result<Load> {
    let Symbol::Function { function, written } = program_context
        .symbols
        .get(name)
        .wrap_err_with(|| format!("Function {name} doesn't exist."))?
    else {
        return Err(eyre!("{name} is not a function."));
    };
    let function = function.clone();
    let written = *written;

    let mut compiler_result = function.compile(program_context)?;
    if !written {
        program_context.text.append(&mut compiler_result);
    }

    let mut loads = Vec::new();
    for element in args.iter().map(|arg| load(arg, program_context)) {
        loads.push(element.wrap_err("In function parameter.")?)
    }

    let mut code_pre = move_all(loads);
    code_pre.push(Executable::Call {
        label: name.to_string(),
    });

    Ok(Load {
        code_pre,
        returns: Loadable::Register(Register::Rax),
        code_post: vec![],
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
            let mut load = build_function(function_name, args, program_context)?;
            let mut vec = load.code_pre;
            vec.append(&mut load.code_post);
            Ok(vec)
        }
        Expression::MathematicalOperation {
            left_side,
            right_side,
            operator: _,
        } => {
            let mut vec = resolve_standalone_expression(left_side, program_context)?;
            let mut rhs = resolve_standalone_expression(right_side, program_context)?;
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
    fn assembly_function(&mut self, name: &str, executable: &Vec<Executable>) -> Function {
        self.binary_function(name, self.as_binary(executable))
    }

    fn binary_function(&mut self, name: &str, binary: Vec<UnlinkedTextSectionElement>) -> Function {
        Function::new(name.to_string(), Code::Binary(binary))
    }

    fn load_onto_stack(&self, loadable: &Loadable) -> Vec<UnlinkedTextSectionElement> {
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

    fn load_into_register(
        &self,
        loadable: &Loadable,
        dest: &Register,
    ) -> Vec<UnlinkedTextSectionElement> {
        match loadable {
            Loadable::Stack => self.one_as_binary(&Executable::Pop(*dest)),
            Loadable::Register(src) => {
                // MOV r64, r/m64
                // REX.W + 8B /r
                vec![UnlinkedTextSectionElement::Binary(vec![
                    Rex::registers_64(src, dest),
                    0x8B,
                    ModRM::register(dest, src),
                ])]
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
                let mut vec = vec![UnlinkedTextSectionElement::Binary(vec![
                    Rex::byte(true, extended(dest), false, false),
                    0xB8 + dest.byte(),
                ])];
                vec.push(UnlinkedTextSectionElement::Binary(x.to_le_bytes().to_vec()));
                vec
            }
        }
    }

    fn one_as_binary(&self, executable: &Executable) -> Vec<UnlinkedTextSectionElement> {
        let bin = match executable {
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
                (loadable, Loadable::Stack) => return self.load_onto_stack(loadable),
                (loadable, Loadable::Register(dest)) => {
                    return self.load_into_register(loadable, dest)
                }
            },
            // CALL rel32
            Executable::Call { label } => {
                return vec![UnlinkedTextSectionElement::Call {
                    function_name: label.clone(),
                }]
            }

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
        };
        vec![UnlinkedTextSectionElement::Binary(bin)]
    }

    fn as_binary(&self, executable: &Vec<Executable>) -> Vec<UnlinkedTextSectionElement> {
        executable
            .into_iter()
            .flat_map(|executable| self.one_as_binary(executable))
            .collect()
    }
}

fn basic_functions(program_context: &mut Program) -> Result<()> {
    let functions = [
        program_context.assembly_function(
            "putchar",
            &vec![
                Executable::MoveLoad {
                    src: Loadable::Register(Register::Rdi),
                    dest: Loadable::Stack,
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
                    src: Loadable::Register(Register::Rsp),
                    dest: Loadable::Register(Register::Rsi),
                },
                Executable::MoveLoad {
                    src: Loadable::Immediate(1),
                    dest: Loadable::Register(Register::Rdx),
                },
                Executable::Syscall,
                Executable::Pop(Register::Rax),
                Executable::Ret,
            ],
        ),
        program_context.assembly_function(
            "exit",
            &vec![
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
            ],
        ),
    ];

    for function in functions {
        program_context.symbols.insert(
            function.ident.clone(),
            Symbol::Function {
                function,
                written: false,
            },
        );
    }

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

pub fn compile_main(mut ast: Ast) -> Result<Vec<u8>> {
    ast.statements.push(Statement::Expression {
        value: Expression::FunctionCall {
            function_name: "exit".to_string(),
            arguments: vec![],
        },
    });

    let main = Function {
        ident: "main".to_string(),
        code: Code::Ast(ast),
        symbols: HashMap::new(),
    };

    let mut program_context = Program::new();
    basic_functions(&mut program_context)?;

    let compiled = main.compile(&mut program_context)?;
    Ok(generate_elf(
        link(compiled).wrap_err("Linker failed.")?,
        Vec::new(),
    ))
}

impl Function {
    fn new(ident: String, code: Code) -> Self {
        Function {
            ident,
            code,
            symbols: HashMap::new(),
        }
    }

    pub fn compile(&self, program_context: &mut Program) -> Result<Vec<Label>> {
        let mut functions_within = HashMap::new();

        let ast = match &self.code {
            Code::Ast(ast) => ast,
            Code::Binary(binary) => {
                return Ok(vec![Label {
                    ident: self.ident.clone(),
                    code: binary.clone(),
                }])
            }
        };

        for statement in &ast.statements {
            if let Statement::FunctionDefinition {
                function_name,
                parameters,
                code,
            } = statement
            {
                let function = Function {
                    code: Code::Ast(code.clone()),
                    ident: function_name.clone(),
                    symbols: load_all(parameters.clone()),
                };
                functions_within.insert(function_name.clone(), function);
            }
        }

        for statement in &ast.statements {
            let executable = match statement {
                Statement::Expression { value } => {
                    resolve_standalone_expression(&value, program_context)
                        .wrap_err("Failed to resolve standalone expression.")?
                }
                _ => todo!(),
            };
            debug!("Generated high level assembly: {executable:?}");
            program_context.text.push(Label {
                ident: self.ident.clone(),
                code: program_context.as_binary(&executable),
            })
        }

        Ok(program_context.text.clone())
    }
}
