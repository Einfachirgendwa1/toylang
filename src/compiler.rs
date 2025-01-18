use crate::{elf::generate_elf, link, Ast, Code, Expression, Impossible, Statement};

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

fn load(expression: &Expression, program: &mut Program) -> Result<Load> {
    let load = match expression {
        Expression::LiteralInt { value } => Load::simple(Loadable::Immediate(*value)),
        Expression::Variable { name } => {
            let Symbol::Variable(_var) = program
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
        } => build_function(function_name, arguments, program)?,
        _ => todo!(),
    };

    Ok(load)
}

fn build_function(name: &str, args: &[Expression], program: &mut Program) -> Result<Load> {
    program.write_function(name)?;

    let mut loads = Vec::new();
    for element in args.iter().map(|arg| load(arg, program)) {
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
    program: &mut Program,
) -> Result<Vec<Executable>> {
    match value {
        Expression::FunctionCall {
            function_name,
            arguments: args,
        } => {
            let mut load = build_function(function_name, args, program)?;
            let mut vec = load.code_pre;
            vec.append(&mut load.code_post);
            Ok(vec)
        }
        Expression::MathematicalOperation {
            left_side,
            right_side,
            ..
        } => {
            let mut vec = resolve_standalone_expression(left_side, program)?;
            let mut rhs = resolve_standalone_expression(right_side, program)?;
            vec.append(&mut rhs);
            Ok(vec)
        }
        Expression::CodeBlock { ast } => ast.compile(program, "<anonymous code block>"),

        expr => {
            warn!("Unused expression: {expr}. Refusing to compile.");
            Ok(Vec::new())
        }
    }
}

const fn extended(register: &Register) -> bool {
    register.byte() >= 8
}

// Source for the assembly: https://www.felixcloutier.com/x86
impl Program {
    fn write_function(&mut self, name: &str) -> Result<()> {
        let Symbol::Function { function, written } = unsafe { &*(self as *const Self) }
            .symbols
            .get(name)
            .wrap_err_with(|| format!("Function {name} doesn't exist."))?
        else {
            return Err(eyre!("{name} is not a function."));
        };

        if !written {
            let mut compiler_result = function.compile(self)?;
            self.text.append(&mut compiler_result);
        }

        let Symbol::Function { written, .. } = self.symbols.get_mut(name).unwrap() else {
            unreachable!()
        };
        *written = true;

        Ok(())
    }

    fn assembly_function(&mut self, name: &str, executable: &[Executable]) -> Function {
        self.binary_function(name, self.generate_binary(executable))
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

    fn generate_binary(&self, executable: &[Executable]) -> Vec<UnlinkedTextSectionElement> {
        executable
            .iter()
            .flat_map(|executable| self.one_as_binary(executable))
            .collect()
    }
}

fn basic_functions(program: &mut Program) -> Result<Function> {
    let functions = [
        program.assembly_function(
            "putchar",
            &[
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
        program.assembly_function(
            "exit",
            &[
                Executable::MoveLoad {
                    src: Loadable::Immediate(60),
                    dest: Loadable::Register(Register::Rax),
                },
                Executable::MoveLoad {
                    src: Loadable::Immediate(0),
                    dest: Loadable::Register(Register::Rdi),
                },
                Executable::Syscall,
            ],
        ),
    ];

    for function in functions {
        let ident = function.ident.clone();
        program.symbols.insert(
            function.ident.clone(),
            Symbol::Function {
                function,
                written: false,
            },
        );
        program.write_function(&ident).impossible()?;
    }

    let start = program.assembly_function(
        "_start",
        &[
            Executable::Call {
                label: "main".to_string(),
            },
            Executable::Call {
                label: "exit".to_string(),
            },
        ],
    );

    Ok(start)
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
        (1 << 6)
            | (self.w as u8 * (1 << 3))
            | (self.r as u8 * (1 << 2))
            | (self.x as u8 * (1 << 1))
            | (self.b as u8)
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

pub fn compile_main(ast: Ast) -> Result<Vec<u8>> {
    let mut program = Program::new();

    basic_functions(&mut program)?.bake(&mut program)?;

    let main = Function {
        ident: "main".to_string(),
        code: Code::Ast(ast),
        symbols: HashMap::new(),
    };

    main.bake(&mut program)?;

    let linker_output = link(program).wrap_err("Linker failed.")?;
    let elf = generate_elf(linker_output, b"hi".to_vec());
    Ok(elf)
}

impl Ast {
    fn compile(&self, program: &mut Program, name: &str) -> Result<Vec<Executable>> {
        debug!("Compiling `{}`", name);

        let mut executable = Vec::new();
        for statement in &self.statements {
            match statement {
                Statement::Expression { value } => {
                    let mut instructions = resolve_standalone_expression(value, program)
                        .wrap_err("Failed to resolve standalone expression.")?;
                    executable.append(&mut instructions)
                }
                Statement::FunctionDefinition {
                    function_name,
                    parameters,
                    result,
                } => {
                    let statements = vec![
                        Statement::Expression {
                            value: result.clone(),
                        },
                        Statement::Return {
                            value: Expression::Unit,
                        },
                    ];

                    let function = Function {
                        code: Code::Ast(Ast { statements }),
                        ident: function_name.clone(),
                        symbols: load_all(parameters.clone()),
                    };

                    program.symbols.insert(
                        function_name.clone(),
                        Symbol::Function {
                            function,
                            written: false,
                        },
                    );
                }

                x @ Statement::Return { value } => {
                    if *value != Expression::Unit {
                        todo!("{x}");
                    }
                    executable.push(Executable::Ret);
                }
                x => todo!("{x}"),
            };
        }

        Ok(executable)
    }
}

impl Function {
    fn new(ident: String, code: Code) -> Self {
        Function {
            ident,
            code,
            symbols: HashMap::new(),
        }
    }

    pub fn bake(&self, program: &mut Program) -> Result<()> {
        program.symbols.insert(
            self.ident.clone(),
            Symbol::Function {
                function: self.clone(),
                written: false,
            },
        );
        program.write_function(&self.ident)
    }

    pub fn compile(&self, program: &mut Program) -> Result<Vec<Label>> {
        let ast = match &self.code {
            Code::Ast(ast) => ast,
            Code::Binary(binary) => {
                debug!("Appending binary func `{}`", self.ident);
                return Ok(vec![Label {
                    ident: self.ident.clone(),
                    code: binary.clone(),
                }]);
            }
        };

        let executable = ast.compile(program, &self.ident)?;

        Ok(vec![Label {
            ident: self.ident.clone(),
            code: program.generate_binary(&executable),
        }])
    }
}
