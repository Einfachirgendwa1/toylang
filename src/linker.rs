use std::collections::{HashMap, VecDeque};

use crate::{Impossible, Label, Program, UnlinkedTextSectionElement};

use eyre::Result;
use log::{debug, warn};

pub struct LinkerOutput {
    pub code: Vec<u8>,
    pub symtab: Vec<Elf64Sym>,
    pub strtab: Vec<String>,
    pub symtab_sh_info: u32,
}

pub fn link(program: Program) -> Result<LinkerOutput> {
    let mut res = Vec::new();
    let mut code = VecDeque::from(program.text);

    while let Some(next) = code.pop_front() {
        let ident = next.ident.to_string();
        let mut this_vec = vec![next.clone()];
        while let Some(position) = code.iter().position(|label| label.ident == ident) {
            let element = code.remove(position).impossible().unwrap();
            this_vec.push(element);
        }
        res.push(this_vec);
    }

    let mut code: Vec<Label> = res
        .into_iter()
        .map(|labels| Label {
            ident: labels[0].ident.to_string(),
            code: labels
                .into_iter()
                .flat_map(|label| label.code.clone())
                .collect(),
        })
        .collect();

    debug!("Linker input: {code:?}");

    let start = code.remove(
        code.iter()
            .position(|label| label.ident.as_str() == "_start")
            .impossible_message("_start label not found.")?,
    );

    code.iter_mut()
        .find(|x| x.ident == "main")
        .impossible_message("main function not found.")?
        .code
        .push(UnlinkedTextSectionElement::Binary(vec![0xC3]));

    code.insert(0, start);

    let mut function_map = HashMap::new();

    let mut position = 0;
    for label in &code {
        function_map.insert(label.ident.clone(), position);

        if label.code.is_empty() {
            warn!("Empty label: {}", label.ident)
        }

        for code in &label.code {
            match code {
                UnlinkedTextSectionElement::Binary(binary) => position += binary.len(),
                UnlinkedTextSectionElement::Call { .. } => position += 5,
            }
        }
    }

    let mut symtab = vec![Elf64Sym::null()];
    let mut strtab = vec!["\0".to_string()];
    let mut result = Vec::new();

    let mut idx = symtab.len();

    let symtab_sh_info = (idx + code.len()) as u32;
    for Label { mut ident, code } in code.into_iter() {
        ident.push('\0');

        symtab.push(Elf64Sym::label(
            get_len_sum(&strtab[..idx]) as u32,
            result.len() as u64,
        ));
        strtab.push(ident);

        idx += 1;

        for element in code {
            match element {
                UnlinkedTextSectionElement::Binary(mut binary) => result.append(&mut binary),
                UnlinkedTextSectionElement::Call { ref function_name } => {
                    let Some(address) = function_map.get(function_name) else {
                        panic!("`{function_name}` not in linker hashmap.");
                    };
                    result.push(0xE8);
                    let result_address = *address as i32 - result.len() as i32 - 4;
                    result.extend_from_slice(&result_address.to_le_bytes());
                }
            }
        }
    }

    symtab.push(Elf64Sym::global_label(
        get_len_sum(&strtab[..idx]) as u32,
        0,
    ));
    strtab.push("_start\0".to_string());

    let result = LinkerOutput {
        symtab_sh_info,
        code: result,
        symtab,
        strtab,
    };
    Ok(result)
}

#[repr(C)]
#[derive(Debug)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

impl Elf64Sym {
    fn null() -> Elf64Sym {
        Elf64Sym {
            st_name: 0,
            st_info: 0,
            st_other: 0,
            st_shndx: 0,
            st_value: 0,
            st_size: 0,
        }
    }

    fn label(st_name: u32, st_value: u64) -> Elf64Sym {
        Elf64Sym {
            st_name,
            st_info: 0,
            st_value: st_value + 0x400000,
            st_shndx: 1,
            st_other: 0,
            st_size: 0,
        }
    }

    fn global_label(st_name: u32, st_value: u64) -> Elf64Sym {
        Elf64Sym {
            st_name,
            st_info: 1 << 4,
            st_value: st_value + 0x400000,
            st_shndx: 1,
            st_other: 0,
            st_size: 0,
        }
    }
}

fn get_len_sum<S: AsRef<str>>(vec: &[S]) -> usize {
    vec.iter().map(|s| s.as_ref().len()).sum()
}
