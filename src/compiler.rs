use crate::{Ast, Expression, Impossible, Statement};

use std::{collections::HashMap, io::Write, iter::repeat};

use eyre::{eyre, ContextCompat, Result, WrapErr};
use log::{debug, warn};

type Address = u64;

impl Program {
    fn write_all(&mut self, binary: &[u8]) {
        self.text
            .write_all(binary)
            .expect("Failed to write to buffer.")
    }
}

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
    Call(i32),
    MoveLoad { src: Loadable, dest: Loadable },
    Syscall,
    Ret,
}

#[derive(Clone, Debug)]
pub enum Loadable {
    Register(Register),
    Work(Vec<u8>, Box<Loadable>),
    Immediate(i64),
    Stack,
}

#[derive(Clone, Debug)]
pub enum Symbol {
    Variable(Expression),
    Function(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub code: Ast,
    pub address: Option<Address>,
    pub symbols: HashMap<String, Loadable>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub symbols: HashMap<String, Symbol>,
    pub text: Vec<u8>,
    pub data: Vec<u8>,
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
    let Symbol::Function(function) = program_context
        .symbols
        .get_mut(name)
        .wrap_err_with(|| format!("Function {name} doesn't exist."))?
    else {
        return Err(eyre!("{name} is not a function."));
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
    fn dummy(text: &[u8]) -> Self {
        Function {
            code: Ast {
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

    pub fn write(&mut self, program: &mut Vec<u8>) -> Result<()> {
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
                .write_all(&program_context.as_binary(&executable))
                .unwrap();
        }

        Ok(program_context.text)
    }
}
